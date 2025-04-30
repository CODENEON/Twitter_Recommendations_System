from datetime import datetime, timedelta
import math
import logging
import time

from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.dataframe as dd
import pandas as pd
import numpy as np

from app.models import Hashtag, Tweet, TweetHashtag
from app import db
from sqlalchemy import func, desc

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_dask_client = None

def get_dask_client(n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
    global _dask_client
    
    if _dask_client is None or not _dask_client.status == 'running':
        logger.info("Starting Dask client for parallel processing...")
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        _dask_client = Client(cluster)
        logger.info(f"Dask dashboard available at: {_dask_client.dashboard_link}")
    
    return _dask_client

def close_dask_client():
    global _dask_client
    
    if _dask_client is not None:
        logger.info("Closing Dask client...")
        _dask_client.close()
        _dask_client = None

def update_trending_scores(use_parallel=False, n_workers=None, 
                         threads_per_worker=2, memory_limit='2GB', batch_size=100):
    
    logger.info("Updating trending scores...")
    start_time = time.time()
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    if not use_parallel:
        _update_trending_scores_sequential(week_ago)
        logger.info(f"Sequential trend score update completed in {time.time() - start_time:.2f}s")
        return
    
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        # Get tweet counts for every hashtag in the last week
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
        
        all_hashtags = Hashtag.query.all()
        
        hashtag_count_dict = {h_id: count for h_id, count in hashtag_counts}
        
        logger.info(f"Processing {len(all_hashtags)} hashtags in parallel...")
        
        # Spliting the hashtags into batches for parallel processing
        hashtag_batches = []
        for i in range(0, len(all_hashtags), batch_size):
            hashtag_batches.append(all_hashtags[i:i+batch_size])
        
        logger.info(f"Processing {len(hashtag_batches)} batches of {batch_size} hashtags each")
        
        @delayed
        def process_hashtag_batch(hashtags):
            result = []
            for hashtag in hashtags:
                tweet_count = hashtag_count_dict.get(hashtag.id, 0)
                
                # Calculate recency factor
                hours_since_update = (datetime.utcnow() - hashtag.last_updated).total_seconds() / 3600
                recency_factor = 1.0 + min(1.0, hours_since_update / 24)
                
                trend_score = math.log(tweet_count + 1) * recency_factor
                
                result.append((hashtag.id, trend_score))
            return result
        
        # Create tasks for each batch
        batch_tasks = []
        for batch in hashtag_batches:
            batch_tasks.append(process_hashtag_batch(batch))
        
        batch_results = client.compute(batch_tasks)
        batch_results = client.gather(batch_results)
        
        all_trend_scores = []
        for batch_result in batch_results:
            all_trend_scores.extend(batch_result)
        
        logger.info(f"Updating {len(all_trend_scores)} hashtag trend scores in database...")
        
        # Update hashtag scores in the database
        for hashtag_id, trend_score in all_trend_scores:
            hashtag = Hashtag.query.get(hashtag_id)
            if hashtag:
                hashtag.trend_score = trend_score
                hashtag.last_updated = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Parallel trend score update completed in {time.time() - start_time:.2f}s")
        
    finally:
        pass

def _update_trending_scores_sequential(week_ago):
    
    # Get tweet counts for each hashtag in the last week
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
    
    # same logic as in parallel processing of calculating trend scores
    for hashtag_id, tweet_count in hashtag_counts:
        hashtag = Hashtag.query.get(hashtag_id)
        if hashtag:
            
            hours_since_update = (datetime.utcnow() - hashtag.last_updated).total_seconds() / 3600
            recency_factor = 1.0 + min(1.0, hours_since_update / 24)
            
            hashtag.trend_score = math.log(tweet_count + 1) * recency_factor
            hashtag.last_updated = datetime.utcnow()
    
    db.session.commit()

def get_trending_hashtags(limit=10, use_parallel=False, 
                        n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    
    # trending scores ko update karna pehle
    update_trending_scores(
        use_parallel=use_parallel,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    trending_hashtags = Hashtag.query.order_by(
        Hashtag.trend_score.desc()
    ).limit(limit).all()
    
    return trending_hashtags

def shutdown_parallel_processing():
    close_dask_client()
