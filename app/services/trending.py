import datetime
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dask.distributed import Client, LocalCluster
from dask import delayed
import pandas as pd
import dask.dataframe as dd
import time
from app.models import Hashtag

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
    return get_trending_hashtags_ml(
        limit=limit, 
        use_parallel=use_parallel, 
        n_workers=n_workers, 
        threads_per_worker=threads_per_worker, 
        memory_limit=memory_limit
    )

def get_trending_hashtags_ml(limit=10, use_parallel=False, n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    logger.info("Getting trending hashtags with ML approach...")
    start_time = time.time()
    hashtags = Hashtag.query.all()
    if not hashtags:
        logger.info("No hashtags found in the database")
        return []
    if not use_parallel or len(hashtags) < 100:
        return _get_trending_hashtags_sequential(hashtags, limit)
    client = get_dask_client(n_workers, threads_per_worker, memory_limit)
    try:
        logger.info(f"Processing {len(hashtags)} hashtags in parallel...")

        @delayed
        def extract_features(hashtags_batch):
            features = []
            hashtag_refs = []
            now = datetime.datetime.now(datetime.timezone.utc)
            for hashtag in hashtags_batch:
                tweet_count = hashtag.tweets.count() if hasattr(hashtag, 'tweets') else 0
                hours_since_update = (now - hashtag.last_updated).total_seconds() / 3600 if hashtag.last_updated else 9999
                recency = 1.0 + min(1.0, hours_since_update / 24)
                trend_score = hashtag.trend_score if hasattr(hashtag, 'trend_score') else 0
                features.append([tweet_count, recency, trend_score])
                hashtag_refs.append(hashtag)
            return features, hashtag_refs

        batch_size = 100
        hashtag_batches = [hashtags[i:i+batch_size] for i in range(0, len(hashtags), batch_size)]
        logger.info(f"Processing {len(hashtag_batches)} batches of hashtags")
        batch_tasks = [extract_features(batch) for batch in hashtag_batches]
        batch_results = client.compute(batch_tasks)
        batch_results = client.gather(batch_results)
        all_features = []
        all_hashtag_refs = []
        for features, hashtag_refs in batch_results:
            all_features.extend(features)
            all_hashtag_refs.extend(hashtag_refs)
        X = np.array(all_features)

        @delayed
        def scale_features(X):
            scaler = StandardScaler()
            return scaler.fit_transform(X)

        @delayed
        def prepare_target(hashtag_refs):
            trend_scores = np.array([
                np.log(h.tweets.count() + 1) * (1.0 + min(1.0, (datetime.datetime.now(datetime.timezone.utc) - h.last_updated).total_seconds() / 3600 / 24)) 
                if hasattr(h, 'tweets') and h.tweets.count() > 0 and h.last_updated else 0
                for h in hashtag_refs
            ])
            threshold = np.percentile(trend_scores, 80)
            return (trend_scores >= threshold).astype(int), trend_scores

        @delayed
        def train_and_predict(X_scaled, y):
            if y.sum() == 0 or y.sum() == len(y):
                return X_scaled[:, 2]
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y)
            return model.predict_proba(X_scaled)[:, 1]

        @delayed
        def rank_hashtags(hashtag_refs, probabilities, limit):
            hashtag_probs = list(zip(hashtag_refs, probabilities))
            hashtag_probs.sort(key=lambda x: x[1], reverse=True)
            return [h for h, _ in hashtag_probs[:limit]]

        X_scaled_future = scale_features(X)
        y_future, trend_scores_future = prepare_target(all_hashtag_refs)
        probabilities_future = train_and_predict(X_scaled_future, y_future)
        trending_hashtags_future = rank_hashtags(all_hashtag_refs, probabilities_future, limit)
        trending_hashtags = client.compute(trending_hashtags_future).result()
        logger.info(f"Trending hashtags computation completed in {time.time() - start_time:.2f}s")
        return trending_hashtags
    except Exception as e:
        logger.error(f"Error in parallel trending hashtags: {e}", exc_info=True)
        return _get_trending_hashtags_sequential(hashtags, limit)
    finally:
        close_dask_client()

def _get_trending_hashtags_sequential(hashtags, limit=10):
    now = datetime.datetime.now(datetime.timezone.utc)

    hashtag_scores = []

    for hashtag in hashtags:
        if hashtag.last_updated:
            # Ensure the datetime is timezone-aware
            last_updated_aware = hashtag.last_updated.replace(tzinfo=datetime.timezone.utc)
            hours_since_update = (now - last_updated_aware).total_seconds() / 3600
        else:
            hours_since_update = 9999

        recency = 1.0 + min(1.0, hours_since_update / 24)
        trend_score = hashtag.trend_score if hasattr(hashtag, 'trend_score') else 0
        features = [hashtag.tweets.count(), recency, trend_score]
        score = features[0] * features[1] * features[2]  # Calculate a combined score
        hashtag_scores.append((hashtag, score))

    hashtag_scores.sort(key=lambda x: x[1], reverse=True)

    trending_hashtags = [hashtag for hashtag, _ in hashtag_scores[:limit]]
    return trending_hashtags

def shutdown_parallel_processing():
    close_dask_client()
