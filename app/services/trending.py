"""
Trending hashtags module with parallel processing.
Uses Dask to parallelize the trend score calculation for all hashtags.
"""

from datetime import datetime, timedelta
import math
import logging
import time

# Dask imports for parallel processing
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.dataframe as dd
import pandas as pd
import numpy as np

from app.models import Hashtag, Tweet, TweetHashtag
from app import db
from sqlalchemy import func, desc

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

def update_trending_scores(use_parallel=False, n_workers=None, 
                         threads_per_worker=2, memory_limit='2GB', batch_size=100):
    """
    Update the trend scores for all hashtags based on recent usage.
    Uses parallel processing for improved performance when many hashtags are present.
    
    Args:
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        batch_size: Batch size for processing hashtags
    """
    logger.info("Updating trending scores...")
    start_time = time.time()
    
    # Get the timestamp for 7 days ago
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    if not use_parallel:
        # Use the original sequential implementation
        _update_trending_scores_sequential(week_ago)
        logger.info(f"Sequential trend score update completed in {time.time() - start_time:.2f}s")
        return
    
    # Get Dask client for parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        # Get all hashtags and count their usage in the last week
        hashtag_counts = db.session.query(
            Hashtag.id,
            func.count(Tweet.id).label('tweet_count')
        ).join(
            TweetHashtag, Hashtag.id == TweetHashtag.hashtag_id
        ).join(
            Tweet, TweetHashtag.tweet_id == Tweet.id
        ).filter(
            Tweet.timestamp >= week_ago
        ).group_by(
            Hashtag.id
        ).all()
        
        # Get all hashtags - we also need to update those with zero recent usage
        all_hashtags = Hashtag.query.all()
        
        # Organize hashtag counts by ID for easy lookup
        hashtag_count_dict = {h_id: count for h_id, count in hashtag_counts}
        
        # Process hashtags in parallel
        logger.info(f"Processing {len(all_hashtags)} hashtags in parallel...")
        
        # Group hashtags into batches for parallel processing
        hashtag_batches = []
        for i in range(0, len(all_hashtags), batch_size):
            hashtag_batches.append(all_hashtags[i:i+batch_size])
        
        logger.info(f"Processing {len(hashtag_batches)} batches of {batch_size} hashtags each")
        
        # Define function to process a batch of hashtags
        @delayed
        def process_hashtag_batch(hashtags):
            """Process a batch of hashtags in parallel."""
            result = []
            for hashtag in hashtags:
                # Get tweet count for this hashtag (0 if not in the dict)
                tweet_count = hashtag_count_dict.get(hashtag.id, 0)
                
                # Calculate recency factor (1.0 to 2.0) based on last update time
                hours_since_update = (datetime.utcnow() - hashtag.last_updated).total_seconds() / 3600
                recency_factor = 1.0 + min(1.0, hours_since_update / 24)  # Max boost of 2x for 24+ hours
                
                # Calculate trend score
                trend_score = math.log(tweet_count + 1) * recency_factor
                
                result.append((hashtag.id, trend_score))
            return result
        
        # Schedule parallel processing of hashtag batches
        batch_tasks = []
        for batch in hashtag_batches:
            batch_tasks.append(process_hashtag_batch(batch))
        
        # Compute all batches in parallel and gather results
        # FIX: Use gather to get all results at once instead of iterating over futures
        batch_results = client.compute(batch_tasks)
        batch_results = client.gather(batch_results)  # Wait for all results and gather them
        
        # Gather all trend scores
        all_trend_scores = []
        for batch_result in batch_results:  # Now batch_results is a list of results, not futures
            all_trend_scores.extend(batch_result)
        
        # Update hashtags in the database
        logger.info(f"Updating {len(all_trend_scores)} hashtag trend scores in database...")
        
        for hashtag_id, trend_score in all_trend_scores:
            hashtag = Hashtag.query.get(hashtag_id)
            if hashtag:
                hashtag.trend_score = trend_score
                hashtag.last_updated = datetime.utcnow()
        
        # Commit all updates
        db.session.commit()
        
        logger.info(f"Parallel trend score update completed in {time.time() - start_time:.2f}s")
        
    finally:
        # Don't close the client here to allow reuse
        pass

def _update_trending_scores_sequential(week_ago):
    """
    Original sequential implementation of trend score updates.
    
    Args:
        week_ago: Timestamp for 7 days ago
    """
    # Get all hashtags and count their usage in the last week
    hashtag_counts = db.session.query(
        Hashtag.id,
        func.count(Tweet.id).label('tweet_count')
    ).join(
        TweetHashtag, Hashtag.id == TweetHashtag.hashtag_id
    ).join(
        Tweet, TweetHashtag.tweet_id == Tweet.id
    ).filter(
        Tweet.timestamp >= week_ago
    ).group_by(
        Hashtag.id
    ).all()
    
    # Update trend scores based on counts and recency
    for hashtag_id, tweet_count in hashtag_counts:
        hashtag = Hashtag.query.get(hashtag_id)
        if hashtag:
            # Simple trending algorithm:
            # Score = log(count + 1) * recency_factor
            # where recency_factor gives higher weight to more recent activity
            
            # Calculate recency factor (1.0 to 2.0) based on last update time
            hours_since_update = (datetime.utcnow() - hashtag.last_updated).total_seconds() / 3600
            recency_factor = 1.0 + min(1.0, hours_since_update / 24)  # Max boost of 2x for 24+ hours
            
            # Update the trend score
            hashtag.trend_score = math.log(tweet_count + 1) * recency_factor
            hashtag.last_updated = datetime.utcnow()
    
    # Commit all updates
    db.session.commit()

def get_trending_hashtags(limit=10, use_parallel=False, 
                        n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get the top trending hashtags.
    
    Args:
        limit: Maximum number of hashtags to return
        use_parallel: Whether to use parallel processing for updating trend scores
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of Hashtag objects sorted by trend_score
    """
    # First update the trending scores to ensure they're current
    update_trending_scores(
        use_parallel=use_parallel,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    # This query is simple and fast, no need to parallelize
    trending_hashtags = Hashtag.query.order_by(
        Hashtag.trend_score.desc()
    ).limit(limit).all()
    
    return trending_hashtags

def shutdown_parallel_processing():
    """Shut down the Dask client when done with all processing."""
    close_dask_client()