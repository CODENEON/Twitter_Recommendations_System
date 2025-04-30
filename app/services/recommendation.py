import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from dask.distributed import Client, LocalCluster
from dask import delayed
import pandas as pd 
from surprise import Dataset, Reader, SVD 
from surprise.model_selection import train_test_split
from app import db
from app.models import User, Tweet, Hashtag, Follow
from sqlalchemy import func
from generate_clusters import generate_clusters as gen_clusters

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_dask_client = None

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

def get_recommended_tweets(user, limit=5, use_parallel=False, 
                          n_workers=None, threads_per_worker=2, memory_limit='2GB'):

    user_tweets = user.tweets.all()
    
    # If the user has no tweets, return recent popular tweets instead
    if not user_tweets:
        popular_tweets = Tweet.query.order_by(
            Tweet.timestamp.desc()
        ).limit(limit).all()
        return popular_tweets
    
    user_tweet_ids = [tweet.id for tweet in user_tweets]
    
    client = get_dask_client(n_workers, threads_per_worker, memory_limit) if use_parallel else None
    
    try:
        # Content-based filtering
        logger.info("Getting content-based recommendations...")
        start_time = time.time()
        content_recommendations = _get_content_based_recommendations(
            user, user_tweet_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Content-based recommendations completed in {time.time() - start_time:.2f}s")
        
        # Collaborative filtering (Hashtag-based)
        logger.info("Getting collaborative recommendations...")
        start_time = time.time()
        collab_recommendations = _get_collaborative_recommendations(
            user, user_tweet_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Collaborative recommendations completed in {time.time() - start_time:.2f}s")

        # Latent Factorization (SVD)
        logger.info("Getting latent factor recommendations...")
        start_time = time.time()
        latent_recommendations = _get_latent_factor_recommendations(
            user, user_tweet_ids, limit*2
        )
        logger.info(f"Latent factor recommendations completed in {time.time() - start_time:.2f}s")
        
        # Combine recommendations, Weighted hybrid approach
        scored_recommendations = {}
        processed_tweet_ids = set(user_tweet_ids)

        weights = {
            'latent': 3,
            'content': 2,
            'collaborative': 1
        }

        def add_scored_recommendations(recommendations, score):
            for tweet in recommendations:
                if tweet.id not in processed_tweet_ids:
                    if tweet.id not in scored_recommendations or score > scored_recommendations[tweet.id][0]:
                        scored_recommendations[tweet.id] = (score, tweet) 
                    # Add to processed set regardless, so we don't pick it from lower-priority sources later
                    processed_tweet_ids.add(tweet.id)

        add_scored_recommendations(latent_recommendations, weights['latent'])
        add_scored_recommendations(content_recommendations, weights['content'])
        add_scored_recommendations(collab_recommendations, weights['collaborative'])

        sorted_scored_tweets = sorted(scored_recommendations.values(), key=lambda item: item[0], reverse=True)

        final_recommendations = [tweet for score, tweet in sorted_scored_tweets]

        if len(final_recommendations) < limit:
            needed = limit - len(final_recommendations)
            logger.info(f"Filling remaining {needed} slots with popular tweets...")
            # Fetch popular tweets not in the processed_tweet_ids set
            popular_tweets = Tweet.query.filter(
                ~Tweet.id.in_(processed_tweet_ids) # Exclude user tweets and all previously considered recommendations
            ).order_by(
                Tweet.timestamp.desc()
            ).limit(needed).all()
            final_recommendations.extend(popular_tweets)

        return final_recommendations[:limit]
        
    finally:
        pass

def _get_content_based_recommendations(user, excluded_tweet_ids, limit=10, 
                                     use_parallel=False, client=None):
    user_tweets = user.tweets.all()
    
    other_tweets = Tweet.query.filter(
        Tweet.user_id != user.id,
        ~Tweet.id.in_(excluded_tweet_ids)
    ).order_by(
        Tweet.timestamp.desc()
    ).limit(500).all() 
    
    if not other_tweets:
        return []
    
    # Combine user tweets and other tweets for vectorization
    all_tweets = user_tweets + other_tweets
    tweet_texts = [tweet.text for tweet in all_tweets]
    
    if use_parallel:
        if client is None:
            client = get_dask_client()
        
        return _parallel_content_recommendations(
            user_tweets, other_tweets, tweet_texts, limit, client
        )
    else:
        return _sequential_content_recommendations(
            user_tweets, other_tweets, tweet_texts, limit
        )

def _parallel_content_recommendations(user_tweets, other_tweets, tweet_texts, limit, client):
    try:
        @delayed
        def compute_tfidf():
            vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
            return vectorizer.fit_transform(tweet_texts)
        
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
        

        @delayed
        def compute_similarities(profile_and_tfidf):
            user_profile, other_tfidf = profile_and_tfidf
            
            similarities = cosine_similarity(user_profile, other_tfidf)[0]
            
            similar_indices = similarities.argsort()[-limit:][::-1]
            
            return [other_tweets[i] for i in similar_indices]
        
        tfidf_matrix = compute_tfidf()
        profile_and_tfidf = compute_user_profile(tfidf_matrix)
        recommendations = compute_similarities(profile_and_tfidf)
        
        result = client.compute(recommendations)
        return result.result() 
    
    except Exception as e:
        logger.error(f"Error in parallel content recommendations: {e}")
        return _sequential_content_recommendations(user_tweets, other_tweets, tweet_texts, limit)

def _sequential_content_recommendations(user_tweets, other_tweets, tweet_texts, limit):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = vectorizer.fit_transform(tweet_texts)
    
    user_tweet_count = len(user_tweets)
    user_profile = np.zeros((1, tfidf_matrix.shape[1]))
    for i in range(user_tweet_count):
        user_profile += tfidf_matrix[i].toarray()
    
    if user_tweet_count > 0:
        user_profile = user_profile / user_tweet_count
    
    other_tfidf = tfidf_matrix[user_tweet_count:]
    similarities = cosine_similarity(user_profile, other_tfidf)[0]
    
    similar_indices = similarities.argsort()[-limit:][::-1]
    
    return [other_tweets[i] for i in similar_indices]


def _get_latent_factor_recommendations(user, excluded_tweet_ids, limit=10):
    try:
        all_user_tweet_pairs = db.session.query(Tweet.user_id, Tweet.id).limit(30000).all()
        if not all_user_tweet_pairs:
            logger.warning("No user-tweet interactions found for SVD.")
            return []

        ratings_df = pd.DataFrame(all_user_tweet_pairs, columns=['userID', 'itemID'])
        ratings_df['rating'] = 1.0 

        reader = Reader(rating_scale=(1, 1)) 

        # Load data into Surprise Dataset
        data = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)

        # Use the full dataset for training (no train-test split needed for prediction)
        trainset = data.build_full_trainset()

        # Training the SVD model
        logger.info("Training SVD model...")
        algo = SVD(n_factors=50, n_epochs=20, random_state=42) 
        algo.fit(trainset)
        logger.info("SVD model training complete.")

        all_tweet_ids = {item_iid for item_iid in trainset.all_items()}
        user_authored_tweets = {trainset.to_raw_iid(item_inner_id) 
                                for (item_inner_id, _) in trainset.ur[trainset.to_inner_uid(user.id)]}
        
        tweets_to_predict = list(all_tweet_ids - user_authored_tweets)
        
        # Filter out tweets already excluded (e.g., user's own recent tweets)
        tweets_to_predict = [tid for tid in tweets_to_predict if tid not in excluded_tweet_ids]

        if not tweets_to_predict:
            logger.info(f"No new tweets found to recommend for user {user.id} via SVD.")
            return []

        predictions = [algo.predict(user.id, tweet_id) for tweet_id in tweets_to_predict]

        predictions.sort(key=lambda x: x.est, reverse=True)
        
        top_n_predictions = predictions[:limit]
        recommended_tweet_ids = [pred.iid for pred in top_n_predictions]

        if not recommended_tweet_ids:
            return []
            
        recommended_tweets = Tweet.query.filter(Tweet.id.in_(recommended_tweet_ids)).all()
        
        # Maintain order based on prediction score
        tweet_map = {tweet.id: tweet for tweet in recommended_tweets}
        ordered_recommendations = [tweet_map[tid] for tid in recommended_tweet_ids if tid in tweet_map]

        logger.info(f"Generated {len(ordered_recommendations)} SVD recommendations for user {user.id}.")
        return ordered_recommendations

    except Exception as e:
        logger.error(f"Error in latent factor recommendations: {e}", exc_info=True)
        return [] 


def _get_collaborative_recommendations(user, excluded_tweet_ids, limit=10, 
                                     use_parallel=False, client=None):
    # Get hashtags used by the current user
    user_hashtags = set()
    for tweet in user.tweets:
        for hashtag in tweet.hashtags:
            user_hashtags.add(hashtag.id)
    
    if not user_hashtags:
        return []
    
    if not use_parallel:
        return _sequential_collaborative_recommendations(
            user, user_hashtags, excluded_tweet_ids, limit
        )
    
    if client is None:
        client = get_dask_client()
    
    try:
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
        
        similar_users = similar_users_query.all()
        similar_user_ids = [u[0] for u in similar_users]
        
        if not similar_user_ids:
            return []
        
        @delayed
        def get_tweets_from_similar_users():
            return Tweet.query.filter(
                Tweet.user_id.in_(similar_user_ids),
                ~Tweet.id.in_(excluded_tweet_ids)
            ).order_by(
                Tweet.timestamp.desc()
            ).limit(limit).all()
        
        recommendations = client.compute(get_tweets_from_similar_users())
        
        return recommendations.result()
    
    finally:
        pass

def _sequential_collaborative_recommendations(user, user_hashtags, excluded_tweet_ids, limit):
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
    
    recommendations = Tweet.query.filter(
        Tweet.user_id.in_(similar_user_ids),
        ~Tweet.id.in_(excluded_tweet_ids)
    ).order_by(
        Tweet.timestamp.desc()
    ).limit(limit).all()
    
    return recommendations

def get_recommended_users(user, limit=7, use_parallel=False,
                         n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    try:
        # Run clustering and capture assignments
        cluster_assignments = {}
        try:
            clusters, user_data = gen_clusters(return_assignments=True)
    
            for idx, user in enumerate(user_data):
                cluster_assignments[user['id']] = int(clusters[idx])
        except Exception as e:
            cluster_assignments = {}
        user_cluster_map = cluster_assignments
        for cluster_id, data in clusters['cluster_analysis'].items():
            for uid in data['user_ids']:
                user_cluster_map[uid] = int(cluster_id)
    except Exception as e:
        user_cluster_map = {}

    # Get users the current user already follows
    followed_user_ids = [follow.followee_id for follow in user.followed]
    followed_user_ids.append(user.id) 

    client = get_dask_client(n_workers, threads_per_worker, memory_limit) if use_parallel else None

    try:
        # Strategy 1: Find users with similar hashtags
        logger.info("Getting hashtag-based user recommendations...")
        start_time = time.time()
        hashtag_based_recommendations = _get_hashtag_based_user_recommendations(
            user, followed_user_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Hashtag-based recommendations completed in {time.time() - start_time:.2f}s")
        
        # Strategy 2: Find "friends of friends (Network Analysis)"
        logger.info("Getting network-based user recommendations...")
        start_time = time.time()
        network_recommendations = _get_network_based_user_recommendations(
            user, followed_user_ids, limit*2, use_parallel=use_parallel, client=client
        )
        logger.info(f"Network-based recommendations completed in {time.time() - start_time:.2f}s")

        # Strategy 3: Cluster-based recommendations
        cluster_recommendations = []
        if user.id in user_cluster_map:
            user_cluster = user_cluster_map[user.id]
            cluster_user_ids = [uid for uid, cid in user_cluster_map.items() if cid == user_cluster and uid not in followed_user_ids]
            if cluster_user_ids:
                cluster_recommendations = User.query.filter(User.id.in_(cluster_user_ids)).limit(limit*2).all()

        combined_recommendations = []
        existing_ids = set()

        # Prioritize cluster-based recommendations
        for rec_user in cluster_recommendations:
            if rec_user.id not in existing_ids and len(combined_recommendations) < limit:
                combined_recommendations.append(rec_user)
                existing_ids.add(rec_user.id)
        # Add hashtag-based recommendations
        for rec_user in hashtag_based_recommendations:
            if rec_user.id not in existing_ids and len(combined_recommendations) < limit:
                combined_recommendations.append(rec_user)
                existing_ids.add(rec_user.id)
        # Add network-based recommendations if there's room
        for rec_user in network_recommendations:
            if rec_user.id not in existing_ids and len(combined_recommendations) < limit:
                combined_recommendations.append(rec_user)
                existing_ids.add(rec_user.id)
        return combined_recommendations[:limit]
    finally:
        pass

def _get_hashtag_based_user_recommendations(user, excluded_user_ids, limit=10,
                                          use_parallel=False, client=None):
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
    
    if not use_parallel:
        return _sequential_hashtag_recommendations(user_hashtags, excluded_user_ids, limit)
    
    if client is None:
        client = get_dask_client()
    
    try:
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

            return [u[0] for u in similar_users]
        
        recommendations = client.compute(find_similar_users())
        
        return recommendations.result()
    
    finally:
        pass

def _sequential_hashtag_recommendations(user_hashtags, excluded_user_ids, limit):
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
    
    return [u[0] for u in similar_users]

def _get_network_based_user_recommendations(user, excluded_user_ids, limit=10,
                                         use_parallel=False, client=None):
    # Get IDs of users that the current user follows
    followed_ids = [follow.followee_id for follow in user.followed]
    
    if not followed_ids:
        return []
    
    if not use_parallel:
        return _sequential_network_recommendations(followed_ids, excluded_user_ids, limit)
    
    if client is None:
        client = get_dask_client()
    
    try:
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
            
            return [u[0] for u in friends_of_friends]
        
        recommendations = client.compute(find_friends_of_friends())
        
        return recommendations.result()
    
    finally:
        pass

def _sequential_network_recommendations(followed_ids, excluded_user_ids, limit):
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
    
    return [u[0] for u in friends_of_friends]

def shutdown_parallel_processing():
    close_dask_client()