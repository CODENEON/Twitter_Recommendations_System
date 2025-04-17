from datetime import datetime, timedelta
import math
from app.models import Hashtag, Tweet, TweetHashtag
from app import db
from sqlalchemy import func, desc

def update_trending_scores():
    """
    Update the trend scores for all hashtags based on recent usage.
    This should be called periodically (e.g., by a scheduler or cron job).
    """
    # Get the timestamp for 7 days ago
    week_ago = datetime.utcnow() - timedelta(days=7)
    
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

def get_trending_hashtags(limit=10):
    """
    Get the top trending hashtags.
    
    Args:
        limit: Maximum number of hashtags to return
        
    Returns:
        List of Hashtag objects sorted by trend_score
    """
    # First update the trending scores to ensure they're current
    update_trending_scores()
    
    # Get the top hashtags by trend_score
    trending_hashtags = Hashtag.query.order_by(
        Hashtag.trend_score.desc()
    ).limit(limit).all()
    
    return trending_hashtags