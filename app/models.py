from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

# Association table for followers
class Follow(db.Model):
    __tablename__ = 'follows'
    follower_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    followee_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Association table for tweet hashtags
class TweetHashtag(db.Model):
    __tablename__ = 'tweet_hashtags'
    tweet_id = db.Column(db.Integer, db.ForeignKey('tweets.id'), primary_key=True)
    hashtag_id = db.Column(db.Integer, db.ForeignKey('hashtags.id'), primary_key=True)

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    email = db.Column(db.String(120), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    bio = db.Column(db.String(160))
    member_since = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    tweets = db.relationship('Tweet', backref='author', lazy='dynamic')
    
    # Follow relationships
    followed = db.relationship('Follow',
                               foreign_keys=[Follow.follower_id],
                               backref=db.backref('follower', lazy='joined'),
                               lazy='dynamic',
                               cascade='all, delete-orphan')
    followers = db.relationship('Follow',
                                foreign_keys=[Follow.followee_id],
                                backref=db.backref('followee', lazy='joined'),
                                lazy='dynamic',
                                cascade='all, delete-orphan')
    
    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def follow(self, user):
        if not self.is_following(user):
            f = Follow(follower=self, followee=user)
            db.session.add(f)
    
    def unfollow(self, user):
        f = self.followed.filter_by(followee_id=user.id).first()
        if f:
            db.session.delete(f)
    
    def is_following(self, user):
        return self.followed.filter_by(followee_id=user.id).first() is not None
    
    def is_followed_by(self, user):
        return self.followers.filter_by(follower_id=user.id).first() is not None
    
    def followed_tweets(self):
        """Get tweets from followed users."""
        return Tweet.query.join(Follow, Follow.followee_id == Tweet.user_id)\
            .filter(Follow.follower_id == self.id)
    
    def get_recent_tweets(self, limit=10):
        """Get the user's most recent tweets."""
        return self.tweets.order_by(Tweet.timestamp.desc()).limit(limit).all()
    
    def __repr__(self):
        return f'<User {self.username}>'


class Tweet(db.Model):
    __tablename__ = 'tweets'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(280))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    sentiment_score = db.Column(db.Float, default=0.0)  # -1.0 to 1.0 for negative to positive
    sentiment_label = db.Column(db.String(10), default='neutral')  # 'positive', 'negative', or 'neutral'
    
    # Relationships
    hashtags = db.relationship('Hashtag', 
                               secondary='tweet_hashtags',
                               backref=db.backref('tweets', lazy='dynamic'),
                               lazy='dynamic')
    
    def add_hashtag(self, hashtag):
        """Add a hashtag to this tweet."""
        if not self.has_hashtag(hashtag):
            self.hashtags.append(hashtag)
    
    def has_hashtag(self, hashtag):
        """Check if this tweet has a specific hashtag."""
        return self.hashtags.filter(Hashtag.id == hashtag.id).first() is not None
    
    def __repr__(self):
        return f'<Tweet {self.id}>'


class Hashtag(db.Model):
    __tablename__ = 'hashtags'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(50), unique=True, index=True)
    trend_score = db.Column(db.Float, default=0.0)  # Higher means more trending
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, text):
        self.text = text.lower()
    
    def __repr__(self):
        return f'<Hashtag {self.text}>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))