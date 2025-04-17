from textblob import TextBlob
from app.models import Tweet

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text using TextBlob.
    Returns a tuple of (sentiment_score, sentiment_label)
    """
    analysis = TextBlob(text)
    
    # Get the polarity score (-1.0 to 1.0)
    score = analysis.sentiment.polarity
    
    # Classify the sentiment based on the score
    if score < -0.1:
        label = 'negative'
    elif score > 0.1:
        label = 'positive'
    else:
        label = 'neutral'
    
    return score, label

def get_user_sentiment_stats(user):
    """
    Calculate sentiment statistics for a user's tweets.
    Returns a dictionary containing counts and percentages.
    """
    # Get all tweets by the user
    tweets = user.tweets.all()
    
    if not tweets:
        # Return default values if user has no tweets
        return {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_count': 0,
            'positive_percent': 0,
            'negative_percent': 0,
            'neutral_percent': 0,
            'average_score': 0.0
        }
    
    # Count tweets by sentiment
    positive_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'positive')
    negative_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'negative')
    neutral_count = sum(1 for tweet in tweets if tweet.sentiment_label == 'neutral')
    total_count = len(tweets)
    
    # Calculate percentages
    positive_percent = round((positive_count / total_count) * 100, 1)
    negative_percent = round((negative_count / total_count) * 100, 1)
    neutral_percent = round((neutral_count / total_count) * 100, 1)
    
    # Calculate average sentiment score
    average_score = sum(tweet.sentiment_score for tweet in tweets) / total_count
    
    # Return statistics as a dictionary
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': total_count,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'neutral_percent': neutral_percent,
        'average_score': average_score,
        # Add data for chart.js visualization
        'chart_data': {
            'labels': ['Positive', 'Neutral', 'Negative'],
            'datasets': [{
                'data': [positive_count, neutral_count, negative_count],
                'backgroundColor': ['#28a745', '#6c757d', '#dc3545']
            }]
        }
    }