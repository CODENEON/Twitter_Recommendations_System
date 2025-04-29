"""
Tweet and user recommendation module with parallel processing.
Uses Dask to parallelize computationally intensive operations.
"""

from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

# Dask imports for parallel processing
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask import delayed

from app import db
from app.models import User, Tweet, Hashtag, Follow
from sqlalchemy import func, and_

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

def get_recommended_tweets(user, limit=5, use_parallel=False, 
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get tweet recommendations for a user based on content similarity and collaborative filtering.
    Uses parallel processing for improved performance on large datasets.
    
    Args:
        user: The User to get recommendations for
        limit: Maximum number of tweets to return
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of recommended Tweet objects
    """
    # Get all user's tweets for content analysis
    user_tweets = user.tweets.all()
    
    # If the user has no tweets, return recent popular tweets instead
    if not user_tweets:
        popular_tweets = Tweet.query.order_by(
            Tweet.timestamp.desc()
        ).limit(limit).all()
        return popular_tweets
    
    # Get IDs of tweets the user has already seen/created
    user_tweet_ids = [tweet.id for tweet in user_tweets]
    
    # Initialize Dask client if using parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit) if use_parallel else None
    
    try:
        # Content-based filtering
        logger.info("Getting content-based recommendations...")
        start_time = time.time()
        content_recommendations = _get_content_based_recommendations(
            user, user_tweet_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Content-based recommendations completed in {time.time() - start_time:.2f}s")
        
        # Collaborative filtering
        logger.info("Getting collaborative recommendations...")
        start_time = time.time()
        collab_recommendations = _get_collaborative_recommendations(
            user, user_tweet_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Collaborative recommendations completed in {time.time() - start_time:.2f}s")
        
        # Combine recommendations (with priority to content-based)
        combined_recommendations = []
        
        # Add content-based recommendations first
        combined_recommendations.extend(content_recommendations)
        
        # Add collaborative recommendations if there's room
        remaining = limit - len(combined_recommendations)
        if remaining > 0:
            # Add only those collaborative recommendations that aren't already in the list
            existing_ids = [tweet.id for tweet in combined_recommendations]
            for tweet in collab_recommendations:
                if tweet.id not in existing_ids and len(combined_recommendations) < limit:
                    combined_recommendations.append(tweet)
        
        return combined_recommendations[:limit]
        
    finally:
        # Don't close the client here to allow reuse
        pass

def _get_content_based_recommendations(user, excluded_tweet_ids, limit=10, 
                                     use_parallel=False, client=None):
    """
    Helper function for content-based filtering with parallel processing.
    
    Args:
        user: User to get recommendations for
        excluded_tweet_ids: IDs of tweets to exclude
        limit: Maximum number of recommendations
        use_parallel: Whether to use parallel processing
        client: Dask client instance (None = create new client)
        
    Returns:
        List of recommended Tweet objects
    """
    # Get user's tweets for analysis
    user_tweets = user.tweets.all()
    
    # Get a sample of other tweets for comparison
    # (limit to recent tweets for efficiency)
    other_tweets = Tweet.query.filter(
        Tweet.user_id != user.id,
        ~Tweet.id.in_(excluded_tweet_ids)
    ).order_by(
        Tweet.timestamp.desc()
    ).limit(500).all()  # Sample size
    
    if not other_tweets:
        return []
    
    # Combine user tweets and other tweets for vectorization
    all_tweets = user_tweets + other_tweets
    tweet_texts = [tweet.text for tweet in all_tweets]
    
    if use_parallel:
        # Get Dask client if not provided
        if client is None:
            client = get_dask_client()
        
        # Process in parallel
        return _parallel_content_recommendations(
            user_tweets, other_tweets, tweet_texts, limit, client
        )
    else:
        # Process sequentially
        return _sequential_content_recommendations(
            user_tweets, other_tweets, tweet_texts, limit
        )

def _parallel_content_recommendations(user_tweets, other_tweets, tweet_texts, limit, client):
    """Parallel implementation of content-based recommendations."""
    try:
        # Define TF-IDF computation as a delayed task
        @delayed
        def compute_tfidf():
            vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
            return vectorizer.fit_transform(tweet_texts)
        
        # Define user profile computation as a delayed task
        @delayed
        def compute_user_profile(tfidf_matrix):
            user_tweet_count = len(user_tweets)
            tfidf_array = tfidf_matrix.toarray()
            
            # Calculate user profile by averaging user tweet vectors
            user_profile = np.zeros((1, tfidf_array.shape[1]))
            for i in range(user_tweet_count):
                user_profile += tfidf_array[i]
            
            if user_tweet_count > 0:
                user_profile = user_profile / user_tweet_count
                
            return (user_profile, tfidf_array[user_tweet_count:])
        
        # Define similarity computation as a delayed task
        @delayed
        def compute_similarities(profile_and_tfidf):
            # FIX: Properly unpack the tuple after computation
            user_profile, other_tfidf = profile_and_tfidf
            
            similarities = cosine_similarity(user_profile, other_tfidf)[0]
            
            # Get the indices of the most similar tweets
            similar_indices = similarities.argsort()[-limit:][::-1]
            
            # Return the recommended tweets
            return [other_tweets[i] for i in similar_indices]
        
        # Chain the tasks
        tfidf_matrix = compute_tfidf()
        profile_and_tfidf = compute_user_profile(tfidf_matrix)
        recommendations = compute_similarities(profile_and_tfidf)
        
        # Compute the final result
        result = client.compute(recommendations)
        return result.result()  # Wait for result
    
    except Exception as e:
        logger.error(f"Error in parallel content recommendations: {e}")
        # Fallback to sequential processing
        return _sequential_content_recommendations(user_tweets, other_tweets, tweet_texts, limit)

def _sequential_content_recommendations(user_tweets, other_tweets, tweet_texts, limit):
    """Sequential implementation of content-based recommendations."""
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = vectorizer.fit_transform(tweet_texts)
    
    # Calculate user profile by averaging user tweet vectors
    user_tweet_count = len(user_tweets)
    user_profile = np.zeros((1, tfidf_matrix.shape[1]))
    for i in range(user_tweet_count):
        user_profile += tfidf_matrix[i].toarray()
    
    if user_tweet_count > 0:
        user_profile = user_profile / user_tweet_count
    
    # Calculate similarity between user profile and other tweets
    other_tfidf = tfidf_matrix[user_tweet_count:]
    similarities = cosine_similarity(user_profile, other_tfidf)[0]
    
    # Get the indices of the most similar tweets
    similar_indices = similarities.argsort()[-limit:][::-1]
    
    # Return the recommended tweets
    return [other_tweets[i] for i in similar_indices]

def _get_collaborative_recommendations(user, excluded_tweet_ids, limit=10, 
                                     use_parallel=False, client=None):
    """
    Helper function for collaborative-based filtering with parallel processing.
    
    Args:
        user: User to get recommendations for
        excluded_tweet_ids: IDs of tweets to exclude
        limit: Maximum number of recommendations
        use_parallel: Whether to use parallel processing
        client: Dask client instance (None = create new client)
        
    Returns:
        List of recommended Tweet objects
    """
    # Get hashtags used by the current user
    user_hashtags = set()
    for tweet in user.tweets:
        for hashtag in tweet.hashtags:
            user_hashtags.add(hashtag.id)
    
    if not user_hashtags:
        return []
    
    # If not using parallel processing, use the original implementation
    if not use_parallel:
        return _sequential_collaborative_recommendations(
            user, user_hashtags, excluded_tweet_ids, limit
        )
    
    # Get Dask client if not provided
    if client is None:
        client = get_dask_client()
    
    try:
        # Find users who use similar hashtags
        similar_users_query = db.session.query(
            User.id,
            func.count(Hashtag.id).label('shared_hashtags')
        ).join(
            Tweet, User.id == Tweet.user_id
        ).join(
            Tweet.hashtags
        ).filter(
            User.id != user.id,
            Hashtag.id.in_(user_hashtags)
        ).group_by(
            User.id
        ).order_by(
            func.count(Hashtag.id).desc()
        ).limit(10)
        
        # Execute query (not parallelized to avoid database connection issues)
        similar_users = similar_users_query.all()
        similar_user_ids = [u[0] for u in similar_users]
        
        if not similar_user_ids:
            return []
        
        # Define parallel task to get tweets from similar users
        @delayed
        def get_tweets_from_similar_users():
            return Tweet.query.filter(
                Tweet.user_id.in_(similar_user_ids),
                ~Tweet.id.in_(excluded_tweet_ids)
            ).order_by(
                Tweet.timestamp.desc()
            ).limit(limit).all()
        
        # Compute in parallel
        recommendations = client.compute(get_tweets_from_similar_users())
        
        return recommendations.result()
    
    finally:
        # Don't close the client here to allow reuse
        pass

def _sequential_collaborative_recommendations(user, user_hashtags, excluded_tweet_ids, limit):
    """Sequential implementation of collaborative recommendations."""
    # Find users who use similar hashtags
    similar_users = db.session.query(
        User.id,
        func.count(Hashtag.id).label('shared_hashtags')
    ).join(
        Tweet, User.id == Tweet.user_id
    ).join(
        Tweet.hashtags
    ).filter(
        User.id != user.id,
        Hashtag.id.in_(user_hashtags)
    ).group_by(
        User.id
    ).order_by(
        func.count(Hashtag.id).desc()
    ).limit(10).all()
    
    similar_user_ids = [u[0] for u in similar_users]
    
    # Get recent tweets from similar users
    recommendations = Tweet.query.filter(
        Tweet.user_id.in_(similar_user_ids),
        ~Tweet.id.in_(excluded_tweet_ids)
    ).order_by(
        Tweet.timestamp.desc()
    ).limit(limit).all()
    
    return recommendations

def get_recommended_users(user, limit=5, use_parallel=False,
                         n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get user recommendations (who to follow) based on shared interests and network analysis.
    Uses parallel processing for improved performance on large datasets.
    
    Args:
        user: The User to get recommendations for
        limit: Maximum number of users to recommend
        use_parallel: Whether to use parallel processing
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        List of recommended User objects
    """
    # Get users the current user already follows
    followed_user_ids = [follow.followee_id for follow in user.followed]
    followed_user_ids.append(user.id)  # Add the user's own ID
    
    # Initialize Dask client if using parallel processing
    client = get_dask_client(n_workers, threads_per_worker, memory_limit) if use_parallel else None
    
    try:
        # Strategy 1: Find users with similar hashtags
        logger.info("Getting hashtag-based user recommendations...")
        start_time = time.time()
        hashtag_based_recommendations = _get_hashtag_based_user_recommendations(
            user, followed_user_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Hashtag-based recommendations completed in {time.time() - start_time:.2f}s")
        
        # Strategy 2: Find "friends of friends"
        logger.info("Getting network-based user recommendations...")
        start_time = time.time()
        network_recommendations = _get_network_based_user_recommendations(
            user, followed_user_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Network-based recommendations completed in {time.time() - start_time:.2f}s")
        
        # Combine recommendations
        combined_recommendations = []
        existing_ids = set()
        
        # Add hashtag-based recommendations first
        for rec_user in hashtag_based_recommendations:
            if rec_user.id not in existing_ids:
                combined_recommendations.append(rec_user)
                existing_ids.add(rec_user.id)
        
        # Add network-based recommendations if there's room
        remaining = limit - len(combined_recommendations)
        if remaining > 0:
            for rec_user in network_recommendations:
                if rec_user.id not in existing_ids and len(combined_recommendations) < limit:
                    combined_recommendations.append(rec_user)
                    existing_ids.add(rec_user.id)
        
        return combined_recommendations[:limit]
        
    finally:
        # Don't close the client here to allow reuse
        pass

def _get_hashtag_based_user_recommendations(user, excluded_user_ids, limit=10,
                                          use_parallel=False, client=None):
    """
    Helper function for hashtag-based user recommendations with parallel processing.
    
    Args:
        user: User to get recommendations for
        excluded_user_ids: IDs of users to exclude
        limit: Maximum number of recommendations
        use_parallel: Whether to use parallel processing
        client: Dask client instance (None = create new client)
        
    Returns:
        List of recommended User objects
    """
    # Get hashtags used by the current user
    user_hashtags = set()
    for tweet in user.tweets:
        for hashtag in tweet.hashtags:
            user_hashtags.add(hashtag.id)
    
    if not user_hashtags:
        # If user hasn't used any hashtags, fall back to popular users
        popular_users = User.query.filter(
            ~User.id.in_(excluded_user_ids)
        ).order_by(
            func.count(Tweet.id).desc()
        ).join(
            Tweet
        ).group_by(
            User.id
        ).limit(limit).all()
        return popular_users
    
    # If not using parallel processing, use the original implementation
    if not use_parallel:
        return _sequential_hashtag_recommendations(user_hashtags, excluded_user_ids, limit)
    
    # Get Dask client if not provided
    if client is None:
        client = get_dask_client()
    
    try:
        # Define parallel task to find similar users
        @delayed
        def find_similar_users():
            similar_users = db.session.query(
                User,
                func.count(Hashtag.id).label('shared_hashtags')
            ).join(
                Tweet, User.id == Tweet.user_id
            ).join(
                Tweet.hashtags
            ).filter(
                ~User.id.in_(excluded_user_ids),
                Hashtag.id.in_(user_hashtags)
            ).group_by(
                User.id
            ).order_by(
                func.count(Hashtag.id).desc()
            ).limit(limit).all()
            
            # Extract just the User objects
            return [u[0] for u in similar_users]
        
        # Compute in parallel
        recommendations = client.compute(find_similar_users())
        
        return recommendations.result()
    
    finally:
        # Don't close the client here to allow reuse
        pass

def _sequential_hashtag_recommendations(user_hashtags, excluded_user_ids, limit):
    """Sequential implementation of hashtag-based recommendations."""
    similar_users = db.session.query(
        User,
        func.count(Hashtag.id).label('shared_hashtags')
    ).join(
        Tweet, User.id == Tweet.user_id
    ).join(
        Tweet.hashtags
    ).filter(
        ~User.id.in_(excluded_user_ids),
        Hashtag.id.in_(user_hashtags)
    ).group_by(
        User.id
    ).order_by(
        func.count(Hashtag.id).desc()
    ).limit(limit).all()
    
    # Extract just the User objects
    return [u[0] for u in similar_users]

def _get_network_based_user_recommendations(user, excluded_user_ids, limit=10,
                                         use_parallel=False, client=None):
    """
    Helper function for network-based user recommendations with parallel processing.
    
    Args:
        user: User to get recommendations for
        excluded_user_ids: IDs of users to exclude
        limit: Maximum number of recommendations
        use_parallel: Whether to use parallel processing
        client: Dask client instance (None = create new client)
        
    Returns:
        List of recommended User objects
    """
    # Get IDs of users that the current user follows
    followed_ids = [follow.followee_id for follow in user.followed]
    
    if not followed_ids:
        return []
    
    # If not using parallel processing, use the original implementation
    if not use_parallel:
        return _sequential_network_recommendations(followed_ids, excluded_user_ids, limit)
    
    # Get Dask client if not provided
    if client is None:
        client = get_dask_client()
    
    try:
        # Define parallel task to find friends of friends
        @delayed
        def find_friends_of_friends():
            friends_of_friends = db.session.query(
                User,
                func.count(Follow.follower_id).label('follower_count')
            ).join(
                Follow, User.id == Follow.followee_id
            ).filter(
                Follow.follower_id.in_(followed_ids),
                ~User.id.in_(excluded_user_ids)
            ).group_by(
                User.id
            ).order_by(
                func.count(Follow.follower_id).desc()
            ).limit(limit).all()
            
            # Extract just the User objects
            return [u[0] for u in friends_of_friends]
        
        # Compute in parallel
        recommendations = client.compute(find_friends_of_friends())
        
        return recommendations.result()
    
    finally:
        # Don't close the client here to allow reuse
        pass

def _sequential_network_recommendations(followed_ids, excluded_user_ids, limit):
    """Sequential implementation of network-based recommendations."""
    friends_of_friends = db.session.query(
        User,
        func.count(Follow.follower_id).label('follower_count')
    ).join(
        Follow, User.id == Follow.followee_id
    ).filter(
        Follow.follower_id.in_(followed_ids),
        ~User.id.in_(excluded_user_ids)
    ).group_by(
        User.id
    ).order_by(
        func.count(Follow.follower_id).desc()
    ).limit(limit).all()
    
    # Extract just the User objects
    return [u[0] for u in friends_of_friends]

def shutdown_parallel_processing():
    """Shut down the Dask client when done with all processing."""
    close_dask_client()