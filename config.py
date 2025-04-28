import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'twitter_analytics.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flask-Login configuration
    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    
    # App specific settings
    TWEETS_PER_PAGE = 10
    TRENDING_HASHTAGS_COUNT = 10
    RECOMMENDATIONS_COUNT = 5
    
    # NLTK and TextBlob settings
    NLTK_DATA_PATH = os.path.join(basedir, 'nltk_data')