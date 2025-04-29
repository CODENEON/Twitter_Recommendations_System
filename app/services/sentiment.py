"""
Sentiment analysis module with parallel processing.
Uses Dask to parallelize sentiment analysis for multiple tweets.
"""

from textblob import TextBlob
import logging
import time
import pandas as pd

# Dask imports for parallel processing
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask import delayed

from app.models import Tweet, User
from app import db

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Dask client for reuse across functions
_dask_client = None

def get_dask_client(n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get or create a Dask client for parallel processing.
    
    Args:
        n_workers: Number of worker processes (None = auto-detect)
        threads_per_worker: Number of threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Dask client instance
    """
    global _dask_client
    
    if _dask_client is None or not _dask_client.status == 'running':
        logger.info("Starting Dask client for parallel processing...")
        
        # Create a local cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        # Create a client
        _dask_client = Client(cluster)
        logger.info(f"Dask dashboard available at: {_dask_client.dashboard_link}")
    
    return _dask_client

def close_dask_client():
    """Close the Dask client when done."""
    global _dask_client
    
    if _dask_client is not None:
        logger.info("Closing Dask client...")
        _dask_client.close()
        _dask_client = None

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text using TextBlob.
    Returns a tuple of (sentiment_score, sentiment_label)
    """
    analysis = TextBlob(text)
    
    # Get the polarity score (-1.0 to 1.0)
    score = analysis.sentiment.polarity
    
    # Classify the sentiment based on the score
    if score < -0.1:
        label = 'negative'
    elif score > 0.1:
        label = 'positive'
    else:
        label = 'neutral'
    
    return score, label

def analyze_sentiment_batch(texts, use_parallel=True, 
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Analyze sentiment for a batch of texts in parallel.
    
    Args:
        texts: List of text strings to analyze
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of (sentiment_score, sentiment_label) tuples
    """
    if not use_parallel or len(texts) < 10:
        # For small batches, use sequential processing
        return [analyze_sentiment(text) for text in texts]
    
    # Get Dask client for parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        # Create a DataFrame with the texts
        df = pd.DataFrame({'text': texts})
        
        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=min(len(df) // 10 + 1, 100))
        
        # Define function to analyze sentiment for each partition
        @delayed
        def process_partition(partition):
            """Process a partition of texts."""
            results = []
            for text in partition['text']:
                score, label = analyze_sentiment(text)
                results.append((score, label))
            return results
        
        # Apply function to each partition
        partition_results = []
        for partition in dask_df.to_delayed():
            partition_results.append(process_partition(partition))
        
        # Compute results in parallel
        computed_results = client.compute(partition_results)
        
        # Flatten results
        results = []
        for partition_result in computed_results:
            results.extend(partition_result)
        
        return results
    
    finally:
        # Don't close the client here to allow reuse
        pass

def update_tweet_sentiments(tweets=None, batch_size=100, use_parallel=True,
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Update sentiment scores for tweets in the database.
    
    Args:
        tweets: List of Tweet objects to update (None = tweets without sentiment)
        batch_size: Size of batches for processing
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Number of tweets updated
    """
    logger.info("Updating tweet sentiment scores...")
    start_time = time.time()
    
    # If no tweets provided, get tweets without sentiment
    if tweets is None:
        tweets = Tweet.query.filter(
            (Tweet.sentiment_score == None) | 
            (Tweet.sentiment_label == None)
        ).all()
    
    if not tweets:
        logger.info("No tweets found to update")
        return 0
    
    logger.info(f"Found {len(tweets)} tweets to analyze")
    
    # Process in batches
    updated_count = 0
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        
        # Extract texts
        batch_texts = [tweet.text for tweet in batch]
        
        # Analyze sentiment in parallel
        logger.info(f"Analyzing batch {i//batch_size + 1}/{len(tweets)//batch_size + 1} ({len(batch)} tweets)")
        
        batch_sentiments = analyze_sentiment_batch(
            batch_texts, 
            use_parallel=use_parallel,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        # Update tweets in the database
        for j, (score, label) in enumerate(batch_sentiments):
            batch[j].sentiment_score = score
            batch[j].sentiment_label = label
            updated_count += 1
        
        # Commit the batch
        db.session.commit()
        logger.info(f"Updated {updated_count}/{len(tweets)} tweets")
    
    logger.info(f"Sentiment update completed in {time.time() - start_time:.2f}s")
    return updated_count

def get_user_sentiment_stats(user, use_parallel=True,
                           n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Calculate sentiment statistics for a user's tweets with parallel processing.
    
    Args:
        user: User object to analyze
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Dictionary containing counts and percentages
    """
    # Get all tweets by the user
    tweets = user.tweets.all()
    
    if not tweets:
        # Return default values if user has no tweets
        return {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_count': 0,
            'positive_percent': 0,
            'negative_percent': 0,
            'neutral_percent': 0,
            'average_score': 0.0,
            # Add data for chart.js visualization
            'chart_data': {
                'labels': ['Positive', 'Neutral', 'Negative'],
                'datasets': [{
                    'data': [0, 0, 0],
                    'backgroundColor': ['#28a745', '#6c757d', '#dc3545']
                }]
            }
        }
    
    # If any tweets don't have sentiment, analyze them first
    tweets_without_sentiment = [tweet for tweet in tweets 
                              if tweet.sentiment_score is None or tweet.sentiment_label is None]
    
    if tweets_without_sentiment:
        logger.info(f"Analyzing sentiment for {len(tweets_without_sentiment)} tweets without sentiment")
        update_tweet_sentiments(
            tweets=tweets_without_sentiment,
            use_parallel=use_parallel,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
    
    # For simple operations like counting and averaging, parallel processing is often
    # not worth the overhead for a single user's tweets. We'll use Dask for very large
    # tweet collections, but for individual users, sequential is usually faster.
    
    # Count tweets by sentiment
    positive_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'positive')
    negative_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'negative')
    neutral_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'neutral')
    total_count = len(tweets)
    
    # Calculate percentages
    positive_percent = round((positive_count / total_count) * 100, 1)
    negative_percent = round((negative_count / total_count) * 100, 1)
    neutral_percent = round((neutral_count / total_count) * 100, 1)
    
    # Calculate average sentiment score
    average_score = sum(tweet.sentiment_score for tweet in tweets) / total_count
    
    # Return statistics as a dictionary
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': total_count,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'neutral_percent': neutral_percent,
        'average_score': average_score,
        # Add data for chart.js visualization
        'chart_data': {
            'labels': ['Positive', 'Neutral', 'Negative'],
            'datasets': [{
                'data': [positive_count, neutral_count, negative_count],
                'backgroundColor': ['#28a745', '#6c757d', '#dc3545']
            }]
        }
    }

def analyze_tweet_sentiment_trends(timeframe='weekly', use_parallel=True,
                                n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Analyze sentiment trends over time for all tweets.
    
    Args:
        timeframe: Time grouping ('daily', 'weekly', or 'monthly')
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Dictionary with sentiment trend data
    """
    logger.info(f"Analyzing tweet sentiment trends with timeframe={timeframe}...")
    start_time = time.time()
    
    # If not using parallel processing or for small datasets, 
    # use standard SQLAlchemy queries
    if not use_parallel:
        return _analyze_trends_sequential(timeframe)
    
    # Get Dask client for parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        # Query tweets with sentiment data
        tweets = Tweet.query.filter(
            Tweet.sentiment_score != None,
            Tweet.sentiment_label != None
        ).all()
        
        if not tweets:
            logger.info("No tweets with sentiment data found")
            return {
                'timeframes': [],
                'positive': [],
                'neutral': [],
                'negative': [],
                'average_score': []
            }
        
        # Create DataFrame with tweet data
        df = pd.DataFrame([{
            'timestamp': tweet.timestamp,
            'sentiment_score': tweet.sentiment_score,
            'sentiment_label': tweet.sentiment_label
        } for tweet in tweets])
        
        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=min(len(df) // 100 + 1, 20))
        
        # Group by timeframe
        if timeframe == 'daily':
            dask_df['date'] = dask_df['timestamp'].dt.date
        elif timeframe == 'weekly':
            # Extract ISO week
            dask_df['date'] = dask_df['timestamp'].dt.to_period('W').dt.start_time.dt.date
        else:  # monthly
            dask_df['date'] = dask_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        
        # Count by label for each date
        @delayed
        def process_date_groups(df):
            """Process sentiment groups for each date."""
            result = []
            # Group by date and sentiment label
            for date, group in df.groupby('date'):
                # Count tweets by sentiment
                sentiment_counts = group['sentiment_label'].value_counts()
                avg_score = group['sentiment_score'].mean()
                
                result.append({
                    'date': date,
                    'positive': sentiment_counts.get('positive', 0),
                    'neutral': sentiment_counts.get('neutral', 0),
                    'negative': sentiment_counts.get('negative', 0),
                    'average_score': avg_score
                })
            return result
        
        # Apply function to each partition
        date_group_tasks = []
        for partition in dask_df.to_delayed():
            date_group_tasks.append(process_date_groups(partition))
        
        # Compute results in parallel
        computed_results = client.compute(date_group_tasks)
        
        # Flatten and combine results
        date_groups = []
        for result in computed_results:
            date_groups.extend(result)
        
        # Sort by date
        date_groups.sort(key=lambda x: x['date'])
        
        # Format results
        trend_data = {
            'timeframes': [str(group['date']) for group in date_groups],
            'positive': [group['positive'] for group in date_groups],
            'neutral': [group['neutral'] for group in date_groups],
            'negative': [group['negative'] for group in date_groups],
            'average_score': [group['average_score'] for group in date_groups]
        }
        
        logger.info(f"Trend analysis completed in {time.time() - start_time:.2f}s")
        return trend_data
    
    finally:
        # Don't close the client here to allow reuse
        pass

def _analyze_trends_sequential(timeframe):
    """
    Sequential implementation of sentiment trend analysis.
    
    Args:
        timeframe: Time grouping ('daily', 'weekly', or 'monthly')
        
    Returns:
        Dictionary with sentiment trend data
    """
    # Import SQLAlchemy functions for date extraction
    from sqlalchemy import func, case
    
    # Define the date grouping based on timeframe
    if timeframe == 'daily':
        date_group = func.date(Tweet.timestamp)
    elif timeframe == 'weekly':
        # Extract ISO week (this will vary by database)
        date_group = func.date_trunc('week', Tweet.timestamp)
    else:  # monthly
        date_group = func.date_trunc('month', Tweet.timestamp)
    
    # Query for sentiment counts by timeframe
    sentiment_counts = db.session.query(
        date_group.label('date'),
        func.sum(case([(Tweet.sentiment_label == 'positive', 1)], else_=0)).label('positive'),
        func.sum(case([(Tweet.sentiment_label == 'neutral', 1)], else_=0)).label('neutral'),
        func.sum(case([(Tweet.sentiment_label == 'negative', 1)], else_=0)).label('negative'),
        func.avg(Tweet.sentiment_score).label('average_score')
    ).filter(
        Tweet.sentiment_score != None,
        Tweet.sentiment_label != None
    ).group_by(
        date_group
    ).order_by(
        date_group
    ).all()
    
    # Format results
    trend_data = {
        'timeframes': [str(row.date) for row in sentiment_counts],
        'positive': [row.positive for row in sentiment_counts],
        'neutral': [row.neutral for row in sentiment_counts],
        'negative': [row.negative for row in sentiment_counts],
        'average_score': [row.average_score for row in sentiment_counts]
    }
    
    return trend_data

def shutdown_parallel_processing():
    """Shut down the Dask client when done with all processing."""
    close_dask_client()