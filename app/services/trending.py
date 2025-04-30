from datetime import datetime
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.array as da
import pandas as pd
import dask.dataframe as dd
import time

from app.models import Hashtag

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

def get_trending_hashtags(limit=10, use_parallel=False, n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get the top trending hashtags using machine learning.
    
    Args:
        limit: Maximum number of hashtags to return
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of Hashtag objects sorted by trending score
    """
    return get_trending_hashtags_ml(
        limit=limit, 
        use_parallel=use_parallel, 
        n_workers=n_workers, 
        threads_per_worker=threads_per_worker, 
        memory_limit=memory_limit
    )

def get_trending_hashtags_ml(limit=10, use_parallel=False, n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get trending hashtags using machine learning and parallel processing.
    
    Args:
        limit: Maximum number of hashtags to return
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of Hashtag objects predicted to be trending
    """
    logger.info("Getting trending hashtags with ML approach...")
    start_time = time.time()
    
    # Get all hashtags from the database
    hashtags = Hashtag.query.all()
    
    if not hashtags:
        logger.info("No hashtags found in the database")
        return []
    
    # If not using parallel processing or for small datasets (< 100 hashtags)
    if not use_parallel or len(hashtags) < 100:
        return _get_trending_hashtags_sequential(hashtags, limit)
    
    # Get Dask client for parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    
    try:
        logger.info(f"Processing {len(hashtags)} hashtags in parallel...")
        
        # Step 1: Prepare feature data in parallel
        @delayed
        def extract_features(hashtags_batch):
            """Extract features from a batch of hashtags."""
            features = []
            hashtag_refs = []
            now = datetime.utcnow()
            
            for hashtag in hashtags_batch:
                # Extract features for each hashtag
                tweet_count = hashtag.tweets.count() if hasattr(hashtag, 'tweets') else 0
                hours_since_update = (now - hashtag.last_updated).total_seconds() / 3600 if hashtag.last_updated else 9999
                recency = 1.0 + min(1.0, hours_since_update / 24)
                trend_score = hashtag.trend_score if hasattr(hashtag, 'trend_score') else 0
                
                features.append([tweet_count, recency, trend_score])
                hashtag_refs.append(hashtag)
                
            return features, hashtag_refs
        
        # Divide hashtags into batches for parallel processing
        batch_size = 100
        hashtag_batches = []
        for i in range(0, len(hashtags), batch_size):
            hashtag_batches.append(hashtags[i:i+batch_size])
        
        logger.info(f"Processing {len(hashtag_batches)} batches of hashtags")
        
        # Process batches in parallel
        batch_tasks = []
        for batch in hashtag_batches:
            batch_tasks.append(extract_features(batch))
        
        # Compute features in parallel
        batch_results = client.compute(batch_tasks)
        batch_results = client.gather(batch_results)
        
        # Combine results from all batches
        all_features = []
        all_hashtag_refs = []
        
        for features, hashtag_refs in batch_results:
            all_features.extend(features)
            all_hashtag_refs.extend(hashtag_refs)
        
        # Convert to numpy array for ML processing
        X = np.array(all_features)
        
        # Step 2: Scale features
        @delayed
        def scale_features(X):
            """Scale the feature matrix."""
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        
        # Step 3: Prepare target variable based on existing trend scores
        @delayed
        def prepare_target(hashtag_refs):
            """Prepare target variable for training."""
            trend_scores = np.array([h.trend_score if hasattr(h, 'trend_score') else np.log(tweet_count + 1) * recency
 for h in hashtag_refs])
            threshold = np.percentile(trend_scores, 80)  # Top 20% are considered trending
            return (trend_scores >= threshold).astype(int), trend_scores
        
        # Step 4: Train model and predict probabilities
        @delayed
        def train_and_predict(X_scaled, y):
            """Train logistic regression and predict trending probabilities."""
            # Check if we have both classes represented
            if y.sum() == 0 or y.sum() == len(y):
                # If only one class, return the raw features as scores
                return X_scaled[:, 2]  # Return existing trend scores
            
            # Otherwise train a model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y)
            return model.predict_proba(X_scaled)[:, 1]
        
        # Step 5: Sort and select top trending hashtags
        @delayed
        def rank_hashtags(hashtag_refs, probabilities, limit):
            """Rank hashtags by trending probability and return top ones."""
            # Combine hashtags with their probabilities
            hashtag_probs = list(zip(hashtag_refs, probabilities))
            
            # Sort by probability (descending)
            hashtag_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Return the top trending hashtags
            return [h for h, _ in hashtag_probs[:limit]]
        
        # Execute the ML pipeline in parallel
        X_scaled_future = scale_features(X)
        y_future, trend_scores_future = prepare_target(all_hashtag_refs)
        probabilities_future = train_and_predict(X_scaled_future, y_future)
        trending_hashtags_future = rank_hashtags(all_hashtag_refs, probabilities_future, limit)
        
        # Get the final result
        trending_hashtags = client.compute(trending_hashtags_future).result()
        
        logger.info(f"Trending hashtags computation completed in {time.time() - start_time:.2f}s")
        return trending_hashtags
        
    except Exception as e:
        logger.error(f"Error in parallel trending hashtags: {e}", exc_info=True)
        # Fallback to sequential processing
        return _get_trending_hashtags_sequential(hashtags, limit)
    
    finally:
        # Don't close the client here to allow reuse
        pass
        
def _get_trending_hashtags_sequential(hashtags, limit=10):
    """
    Sequential implementation of trending hashtags with ML.
    
    Args:
        hashtags: List of Hashtag objects
        limit: Maximum number of hashtags to return
        
    Returns:
        List of Hashtag objects sorted by trending score
    """
    features = []
    now = datetime.utcnow()
    
    for hashtag in hashtags:
        tweet_count = hashtag.tweets.count() if hasattr(hashtag, 'tweets') else 0
        hours_since_update = (now - hashtag.last_updated).total_seconds() / 3600 if hashtag.last_updated else 9999
        recency = 1.0 + min(1.0, hours_since_update / 24)
        trend_score = hashtag.trend_score if hasattr(hashtag, 'trend_score') else 0
        features.append([tweet_count, recency, trend_score])
    
    if not features:
        return []
    
    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    trend_scores = np.array([h.trend_score if hasattr(h, 'trend_score') else 0 for h in hashtags])
    threshold = np.percentile(trend_scores, 80)
    y = (trend_scores >= threshold).astype(int)
    
    if y.sum() == 0 or y.sum() == len(y):
        # If only one class, sort by existing trend score
        sorted_hashtags = sorted(hashtags, key=lambda h: h.trend_score if hasattr(h, 'trend_score') else 0, reverse=True)
        return sorted_hashtags[:limit]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    hashtag_probs = list(zip(hashtags, probs))
    hashtag_probs.sort(key=lambda x: x[1], reverse=True)
    
    trending = [h for h, p in hashtag_probs[:limit]]
    return trending

def shutdown_parallel_processing():
    """Shut down the Dask client when done with all processing."""
    close_dask_client()