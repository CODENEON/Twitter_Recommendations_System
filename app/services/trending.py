from datetime import datetime
import logging
from dask.distributed import Client, LocalCluster
from app.models import Hashtag
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

def get_trending_hashtags(limit=10, use_parallel=False, n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    return get_trending_hashtags_ml(limit=limit, use_parallel=use_parallel, n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

def get_trending_hashtags_ml(limit=10, use_parallel=False, n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    hashtags = Hashtag.query.all()
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
    # For demonstration, use trend_score as a proxy label (top 20% as trending)
    trend_scores = np.array([h.trend_score if hasattr(h, 'trend_score') else 0 for h in hashtags])
    threshold = np.percentile(trend_scores, 80)
    y = (trend_scores >= threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        # Fallback: just return top by trend_score
        sorted_hashtags = sorted(hashtags, key=lambda h: h.trend_score if hasattr(h, 'trend_score') else 0, reverse=True)
        return sorted_hashtags[:limit]
    model = LogisticRegression()
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:,1]
    hashtag_probs = list(zip(hashtags, probs))
    hashtag_probs.sort(key=lambda x: x[1], reverse=True)
    trending = [h for h, p in hashtag_probs[:limit]]
    return trending

def shutdown_parallel_processing():
    close_dask_client()
