from flask import Blueprint, jsonify, current_app
from flask_login import login_required, current_user
from app.models import User, Tweet, Hashtag, Like, Retweet, Follow
from app import db
from sqlalchemy import func, desc
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

api = Blueprint('api', __name__)

@api.route('/recommendations')
@login_required
def get_recommendations():
    """Get personalized recommendations for the current user"""
    # Get user's recent tweets and hashtags
    recent_tweets = Tweet.query.filter_by(user_id=current_user.id)\
        .order_by(Tweet.created_at.desc())\
        .limit(10)\
        .all()
    
    # Get user's recent hashtags
    recent_hashtags = []
    for tweet in recent_tweets:
        recent_hashtags.extend([tag.name for tag in tweet.hashtags])
    
    # Get similar users based on hashtag usage
    similar_users = find_similar_users(current_user.id)
    
    # Get recommended tweets from similar users
    recommended_tweets = []
    for user_id in similar_users:
        tweets = Tweet.query.filter_by(user_id=user_id)\
            .order_by(Tweet.created_at.desc())\
            .limit(5)\
            .all()
        recommended_tweets.extend(tweets)
    
    # Get recommended hashtags based on user's interests
    recommended_hashtags = get_recommended_hashtags(current_user.id)
    
    return jsonify({
        'recommended_tweets': [tweet.to_dict() for tweet in recommended_tweets],
        'recommended_hashtags': recommended_hashtags
    })

@api.route('/network/analysis')
@login_required
def analyze_network():
    """Analyze the social network"""
    # Get all users and their connections
    users = User.query.all()
    connections = []
    
    for user in users:
        # Get user's followers
        followers = Follow.query.filter_by(followed_id=user.id).all()
        for follow in followers:
            connections.append({
                'source': follow.follower_id,
                'target': user.id,
                'type': 'follow'
            })
    
    # Get all retweets
    retweets = Retweet.query.all()
    for retweet in retweets:
        connections.append({
            'source': retweet.user_id,
            'target': retweet.tweet.user_id,
            'type': 'retweet'
        })
    
    # Get all likes
    likes = Like.query.all()
    for like in likes:
        connections.append({
            'source': like.user_id,
            'target': like.tweet.user_id,
            'type': 'like'
        })
    
    return jsonify({
        'nodes': [user.to_dict() for user in users],
        'links': connections
    })

@api.route('/hashtags/analysis')
@login_required
def analyze_hashtags():
    """Analyze hashtag relationships and trends"""
    # Get all hashtags and their co-occurrences
    hashtags = Hashtag.query.all()
    co_occurrences = {}
    
    for hashtag in hashtags:
        co_occurrences[hashtag.name] = {}
        for tweet in hashtag.tweets:
            for other_hashtag in tweet.hashtags:
                if other_hashtag.name != hashtag.name:
                    if other_hashtag.name not in co_occurrences[hashtag.name]:
                        co_occurrences[hashtag.name][other_hashtag.name] = 0
                    co_occurrences[hashtag.name][other_hashtag.name] += 1
    
    # Get trending hashtags
    trending_hashtags = get_trending_hashtags()
    
    return jsonify({
        'co_occurrences': co_occurrences,
        'trending_hashtags': trending_hashtags
    })

def find_similar_users(user_id, limit=5):
    """Find users with similar interests based on hashtag usage"""
    # Get target user's hashtags
    target_user = User.query.get(user_id)
    target_hashtags = set()
    for tweet in target_user.tweets:
        target_hashtags.update([tag.name for tag in tweet.hashtags])
    
    # Get all other users and their hashtags
    users = User.query.filter(User.id != user_id).all()
    similarities = []
    
    for user in users:
        user_hashtags = set()
        for tweet in user.tweets:
            user_hashtags.update([tag.name for tag in tweet.hashtags])
        
        # Calculate Jaccard similarity
        intersection = len(target_hashtags.intersection(user_hashtags))
        union = len(target_hashtags.union(user_hashtags))
        similarity = intersection / union if union > 0 else 0
        
        similarities.append((user.id, similarity))
    
    # Sort by similarity and return top users
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [user_id for user_id, _ in similarities[:limit]]

def get_recommended_hashtags(user_id, limit=10):
    """Get recommended hashtags based on user's interests"""
    # Get user's recent hashtags
    user = User.query.get(user_id)
    recent_hashtags = set()
    for tweet in user.tweets.order_by(Tweet.created_at.desc()).limit(20):
        recent_hashtags.update([tag.name for tag in tweet.hashtags])
    
    # Get co-occurring hashtags
    co_occurring = Counter()
    for hashtag_name in recent_hashtags:
        hashtag = Hashtag.query.filter_by(name=hashtag_name).first()
        if hashtag:
            for tweet in hashtag.tweets:
                for other_hashtag in tweet.hashtags:
                    if other_hashtag.name not in recent_hashtags:
                        co_occurring[other_hashtag.name] += 1
    
    # Get trending hashtags
    trending = get_trending_hashtags()
    
    # Combine and rank recommendations
    recommendations = {}
    for hashtag, count in co_occurring.items():
        recommendations[hashtag] = count * 2  # Weight co-occurrences higher
    
    for hashtag, count in trending.items():
        if hashtag in recommendations:
            recommendations[hashtag] += count
        else:
            recommendations[hashtag] = count
    
    # Sort and return top recommendations
    sorted_recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [hashtag for hashtag, _ in sorted_recommendations[:limit]]

def get_trending_hashtags(hours=24, limit=10):
    """Get currently trending hashtags"""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    # Get hashtag usage counts
    hashtag_counts = db.session.query(
        Hashtag.name,
        func.count(Tweet.id).label('count')
    ).join(
        Tweet.hashtags
    ).filter(
        Tweet.created_at >= since
    ).group_by(
        Hashtag.name
    ).order_by(
        desc('count')
    ).limit(limit).all()
    
    return {hashtag: count for hashtag, count in hashtag_counts} 