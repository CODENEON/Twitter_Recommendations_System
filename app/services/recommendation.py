from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app import db
from app.models import User, Tweet, Hashtag, Follow
from sqlalchemy import func, and_

def get_recommended_tweets(user, limit=5):
    """
    Get tweet recommendations for a user based on content similarity and collaborative filtering.
    
    Args:
        user: The User to get recommendations for
        limit: Maximum number of tweets to return
        
    Returns:
        List of recommended Tweet objects
    """
    # Strategy:
    # 1. Content-based: Find tweets similar to the user's liked/posted tweets
    # 2. Collaborative filtering: Find tweets liked by similar users
    # 3. Combine and return the top recommendations
    
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
    
    # Content-based filtering
    content_recommendations = _get_content_based_recommendations(user, user_tweet_ids, limit*2)
    
    # Collaborative filtering
    collab_recommendations = _get_collaborative_recommendations(user, user_tweet_ids, limit*2)
    
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

def _get_content_based_recommendations(user, excluded_tweet_ids, limit=10):
    """Helper function for content-based filtering."""
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
    recommendations = [other_tweets[i] for i in similar_indices]
    return recommendations

def _get_collaborative_recommendations(user, excluded_tweet_ids, limit=10):
    """Helper function for collaborative-based filtering."""
    # For this simplified implementation, we'll use a proxy for collaborative filtering:
    # Find users similar to the current user based on hashtag usage,
    # then recommend their recent tweets
    
    # Get hashtags used by the current user
    user_hashtags = set()
    for tweet in user.tweets:
        for hashtag in tweet.hashtags:
            user_hashtags.add(hashtag.id)
    
    if not user_hashtags:
        return []
    
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

def get_recommended_users(user, limit=5):
    """
    Get user recommendations (who to follow) based on shared interests and network analysis.
    
    Args:
        user: The User to get recommendations for
        limit: Maximum number of users to recommend
        
    Returns:
        List of recommended User objects
    """
    # Get users the current user already follows
    followed_user_ids = [follow.followee_id for follow in user.followed]
    followed_user_ids.append(user.id)  # Add the user's own ID
    
    # Strategy 1: Find users with similar hashtag usage
    hashtag_based_recommendations = _get_hashtag_based_user_recommendations(
        user, followed_user_ids, limit
    )
    
    # Strategy 2: Find users with similar sentiment patterns
    sentiment_based_recommendations = _get_sentiment_based_user_recommendations(
        user, followed_user_ids, limit
    )
    
    # Strategy 3: Find users with similar keywords in tweets
    keyword_based_recommendations = _get_keyword_based_user_recommendations(
        user, followed_user_ids, limit
    )
    
    # Strategy 4: Find "friends of friends"
    network_recommendations = _get_network_based_user_recommendations(
        user, followed_user_ids, limit
    )
    
    # Combine recommendations with a weighted approach
    combined_recommendations = _combine_user_recommendations(
        [
            (hashtag_based_recommendations, 0.4),    # 40% weight
            (sentiment_based_recommendations, 0.2),  # 20% weight
            (keyword_based_recommendations, 0.3),    # 30% weight
            (network_recommendations, 0.1)           # 10% weight
        ],
        limit
    )
    
    return combined_recommendations

def _combine_user_recommendations(recommendation_sets, limit):
    """
    Combine different recommendation sets with weights.
    
    Args:
        recommendation_sets: List of tuples (recommendations, weight)
        limit: Maximum recommendations to return
        
    Returns:
        Combined list of recommendations
    """
    # Create a score dictionary for all recommended users
    user_scores = {}
    
    # Process each recommendation set with its weight
    for recommendations, weight in recommendation_sets:
        for i, user in enumerate(recommendations):
            # Calculate score (higher position = higher score)
            position_score = (len(recommendations) - i) / len(recommendations)
            weighted_score = position_score * weight
            
            # Add to user's total score
            if user.id in user_scores:
                user_scores[user.id]['score'] += weighted_score
            else:
                user_scores[user.id] = {
                    'user': user,
                    'score': weighted_score
                }
    
    # Sort by total score and return top recommendations
    sorted_recommendations = sorted(
        user_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    
    return [item['user'] for item in sorted_recommendations[:limit]]

def _get_hashtag_based_user_recommendations(user, excluded_user_ids, limit=10):
    """Helper function for hashtag-based user recommendations."""
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
    
    # Find users who use similar hashtags
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

def _get_sentiment_based_user_recommendations(user, excluded_user_ids, limit=10):
    """Helper function for sentiment-based user recommendations."""
    # Calculate average sentiment score for the current user
    user_tweets = user.tweets.all()
    if not user_tweets:
        return []
    
    # Calculate user's average sentiment
    user_sentiment_scores = [tweet.sentiment_score for tweet in user_tweets]
    user_avg_sentiment = sum(user_sentiment_scores) / len(user_sentiment_scores)
    
    # Get the most common sentiment label for the user
    user_sentiment_labels = [tweet.sentiment_label for tweet in user_tweets]
    user_dominant_sentiment = Counter(user_sentiment_labels).most_common(1)[0][0]
    
    # Find users with similar sentiment patterns
    similar_sentiment_users = db.session.query(
        User,
        func.avg(Tweet.sentiment_score).label('avg_sentiment')
    ).join(
        Tweet, User.id == Tweet.user_id
    ).filter(
        ~User.id.in_(excluded_user_ids)
    ).group_by(
        User.id
    ).having(
        # Find users with average sentiment within a range of the user's average
        func.avg(Tweet.sentiment_score).between(
            user_avg_sentiment - 0.3, user_avg_sentiment + 0.3
        )
    ).order_by(
        # Order by how close they are to the user's average sentiment
        func.abs(func.avg(Tweet.sentiment_score) - user_avg_sentiment)
    ).limit(limit).all()
    
    # Also find users with the same dominant sentiment label
    dominant_sentiment_users = db.session.query(
        User,
        func.count(Tweet.id).label('tweet_count')
    ).join(
        Tweet, User.id == Tweet.user_id
    ).filter(
        ~User.id.in_(excluded_user_ids),
        Tweet.sentiment_label == user_dominant_sentiment
    ).group_by(
        User.id
    ).order_by(
        func.count(Tweet.id).desc()
    ).limit(limit).all()
    
    # Combine both sets, prioritizing those who appear in both
    sentiment_users = []
    
    # Get user IDs from both sets
    similar_avg_ids = [u[0].id for u in similar_sentiment_users]
    similar_label_ids = [u[0].id for u in dominant_sentiment_users]
    
    # First add users who appear in both sets
    for user_id in set(similar_avg_ids).intersection(set(similar_label_ids)):
        for u in similar_sentiment_users:
            if u[0].id == user_id:
                sentiment_users.append(u[0])
                break
    
    # Then add remaining users from both sets
    for users in [similar_sentiment_users, dominant_sentiment_users]:
        for u in users:
            if u[0].id not in [user.id for user in sentiment_users]:
                sentiment_users.append(u[0])
                if len(sentiment_users) >= limit:
                    break
        if len(sentiment_users) >= limit:
            break
    
    return sentiment_users[:limit]

def _get_keyword_based_user_recommendations(user, excluded_user_ids, limit=10):
    """Helper function for keyword-based user recommendations."""
    # Extract text from user's tweets
    user_tweets = user.tweets.all()
    if not user_tweets:
        return []
    
    # Combine all user's tweet texts
    user_tweet_text = " ".join([tweet.text for tweet in user_tweets])
    
    # Extract important keywords using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    
    # We need at least 2 documents for TF-IDF, so add a dummy document
    all_texts = [user_tweet_text, "dummy document"]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Get the feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the TF-IDF scores for the user's combined text
    user_tfidf = tfidf_matrix[0]
    
    # Get the keywords with highest TF-IDF scores
    user_keyword_indices = user_tfidf.toarray()[0].argsort()[-20:][::-1]
    user_keywords = [feature_names[i] for i in user_keyword_indices]
    
    # Find users who use similar keywords
    other_users = User.query.filter(
        ~User.id.in_(excluded_user_ids)
    ).all()
    
    user_similarity_scores = []
    
    # For each user, check how many of the keywords appear in their tweets
    for other_user in other_users:
        other_tweets = other_user.tweets.all()
        if not other_tweets:
            continue
        
        other_tweet_text = " ".join([tweet.text for tweet in other_tweets])
        
        # Count how many of the user's keywords appear in this user's tweets
        keyword_matches = sum(1 for keyword in user_keywords if keyword.lower() in other_tweet_text.lower())
        
        if keyword_matches > 0:
            user_similarity_scores.append((other_user, keyword_matches))
    
    # Sort by number of keyword matches (descending)
    user_similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the User objects
    return [u[0] for u in user_similarity_scores[:limit]]

def _get_network_based_user_recommendations(user, excluded_user_ids, limit=10):
    """Helper function for network-based user recommendations (friends of friends)."""
    # Get IDs of users that the current user follows
    followed_ids = [follow.followee_id for follow in user.followed]
    
    if not followed_ids:
        return []
    
    # Find users followed by the users the current user follows (friends of friends)
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