from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user
from app.models import Tweet, User
from app.services.trending import get_trending_hashtags
from app.services.recommendation import get_recommended_tweets, get_recommended_users
from app.services.sentiment import get_user_sentiment_stats

dashboard = Blueprint('dashboard', __name__)

@dashboard.route('/')
@dashboard.route('/dashboard')
@login_required
def index():
    # Get total tweet count
    total_tweets = Tweet.query.count()
    
    # Get user's recent tweets
    recent_tweets = current_user.get_recent_tweets(
        limit=current_app.config['TWEETS_PER_PAGE']
    )
    
    # Get tweets from followed users
    followed_tweets = current_user.followed_tweets().order_by(
        Tweet.timestamp.desc()
    ).limit(current_app.config['TWEETS_PER_PAGE']).all()
    
    # Get trending hashtags
    trending_hashtags = get_trending_hashtags(
        limit=current_app.config['TRENDING_HASHTAGS_COUNT']
    )
    
    # Get tweet recommendations
    recommended_tweets = get_recommended_tweets(
        current_user, 
        limit=current_app.config['RECOMMENDATIONS_COUNT']
    )
    
    # Get user recommendations (who to follow)
    recommended_users = get_recommended_users(
        current_user,
        limit=current_app.config['RECOMMENDATIONS_COUNT']
    )
    
    # Get sentiment statistics for visualization
    sentiment_stats = get_user_sentiment_stats(current_user)
    
    return render_template('dashboard/index.html',
                          title='Dashboard',
                          total_tweets=total_tweets,
                          recent_tweets=recent_tweets,
                          followed_tweets=followed_tweets,
                          trending_hashtags=trending_hashtags,
                          recommended_tweets=recommended_tweets,
                          recommended_users=recommended_users,
                          sentiment_stats=sentiment_stats)