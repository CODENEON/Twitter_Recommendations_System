from textblob import TextBlob
import logging
import time
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask import delayed

from app.models import Tweet, User
from app import db


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_dask_client = None

# Initialize Dask client
def get_dask_client(n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
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
    global _dask_client
    
    if _dask_client is not None:
        logger.info("Closing Dask client...")
        _dask_client.close()
        _dask_client = None

# It analyzes the sentiment of a text using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    
    # Get the polarity score (-1.0 to 1.0)
    score = analysis.sentiment.polarity
    
    # Classify the sentiment based on the score
    if score < -0.3:
        label = 'negative'
    elif score > 0.3:
        label = 'positive'
    else:
        label = 'neutral'
    
    return score, label

def analyze_sentiment_batch(texts, use_parallel=True, 
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
    
    # small batches don't need parallelizing
    if not use_parallel or len(texts) < 10:
        return [analyze_sentiment(text) for text in texts]
    
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        df = pd.DataFrame({'text': texts})
        
        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=min(len(df) // 10 + 1, 100))
        
        @delayed
        def process_partition(partition):
            results = []
            for text in partition['text']:
                score, label = analyze_sentiment(text)
                results.append((score, label))
            return results
        
        partition_results = []
        for partition in dask_df.to_delayed():
            partition_results.append(process_partition(partition))
        
        computed_results = client.compute(partition_results)
        
        # Flatten results
        results = []
        for partition_result in computed_results:
            results.extend(partition_result)
        
        return results
    
    finally:
        pass

def update_tweet_sentiments(tweets=None, batch_size=100, use_parallel=True,
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
    logger.info("Updating tweet sentiment scores...")
    start_time = time.time()
    
    # If no tweets are provided, fetch all tweets without sentiment
    
    if tweets is None:
        tweets = Tweet.query.filter(
            (Tweet.sentiment_score == None) | 
            (Tweet.sentiment_label == None)
        ).all()
    
    if not tweets:
        logger.info("No tweets found to update")
        return 0
    
    logger.info(f"Found {len(tweets)} tweets to analyze")
    
    updated_count = 0
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        
        # Extracting text from tweets
        batch_texts = [tweet.text for tweet in batch]
        
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
        
        db.session.commit()
        logger.info(f"Updated {updated_count}/{len(tweets)} tweets")
    
    logger.info(f"Sentiment update completed in {time.time() - start_time:.2f}s")
    return updated_count

def get_user_sentiment_stats(user, use_parallel=True,
                           n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
    tweets = user.tweets.all()
    
    # If the user is new to the platform
    if not tweets:
        return {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_count': 0,
            'positive_percent': 0,
            'negative_percent': 0,
            'neutral_percent': 0,
            'average_score': 0.0,
            
            'chart_data': {
                'labels': ['Positive', 'Neutral', 'Negative'],
                'datasets': [{
                    'data': [0, 0, 0],
                    'backgroundColor': ['#28a745', '#6c757d', '#dc3545']
                }]
            }
        }
        
    # Filter tweets that don't have sentiment data
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
    
    positive_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'positive')
    negative_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'negative')
    neutral_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'neutral')
    total_count = len(tweets)
    
    positive_percent = round((positive_count / total_count) * 100, 1)
    negative_percent = round((negative_count / total_count) * 100, 1)
    neutral_percent = round((neutral_count / total_count) * 100, 1)
    
    average_score = sum(tweet.sentiment_score for tweet in tweets) / total_count
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': total_count,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'neutral_percent': neutral_percent,
        'average_score': average_score,
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
    
    logger.info(f"Analyzing tweet sentiment trends with timeframe={timeframe}...")
    start_time = time.time()
    
    if not use_parallel:
        return _analyze_trends_sequential(timeframe)
    
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
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
        
        # Creating DataFrame with tweet data
        df = pd.DataFrame([{
            'timestamp': tweet.timestamp,
            'sentiment_score': tweet.sentiment_score,
            'sentiment_label': tweet.sentiment_label
        } for tweet in tweets])
        
        # Converting to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=min(len(df) // 100 + 1, 20))
        
        if timeframe == 'daily':
            dask_df['date'] = dask_df['timestamp'].dt.date
        elif timeframe == 'weekly':
            dask_df['date'] = dask_df['timestamp'].dt.to_period('W').dt.start_time.dt.date
        else:
            dask_df['date'] = dask_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        
        @delayed
        def process_date_groups(df):
            result = []
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
        
        date_group_tasks = []
        for partition in dask_df.to_delayed():
            date_group_tasks.append(process_date_groups(partition))
        
        computed_results = client.compute(date_group_tasks)
        
        # Flatten and combine results
        date_groups = []
        for result in computed_results:
            date_groups.extend(result)
        
        date_groups.sort(key=lambda x: x['date'])
        
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
        pass

def _analyze_trends_sequential(timeframe):
    
    from sqlalchemy import func, case
    
    if timeframe == 'daily':
        date_group = func.date(Tweet.timestamp)
    elif timeframe == 'weekly':
        date_group = func.date_trunc('week', Tweet.timestamp)
    else:
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
    
    trend_data = {
        'timeframes': [str(row.date) for row in sentiment_counts],
        'positive': [row.positive for row in sentiment_counts],
        'neutral': [row.neutral for row in sentiment_counts],
        'negative': [row.negative for row in sentiment_counts],
        'average_score': [row.average_score for row in sentiment_counts]
    }
    
    return trend_data

def shutdown_parallel_processing():
    close_dask_client()
