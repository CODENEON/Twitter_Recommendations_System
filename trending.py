from datetime import datetime, timedelta
from sqlalchemy import func, desc
from app import db
from app.models import Hashtag, Tweet

def get_trending_hashtags(hours=24, limit=10):
    """
    Get trending hashtags based on usage frequency in the last specified hours.
    
    Args:
        hours (int): Time window in hours to look for trending hashtags (default: 24)
        limit (int): Maximum number of trending hashtags to return (default: 10)
    
    Returns:
        dict: Dictionary of hashtag names and their usage counts
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    
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