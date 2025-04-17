import re
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length
from app import db
from app.models import Tweet, Hashtag
from app.services.sentiment import analyze_sentiment

tweet = Blueprint('tweet', __name__)

class TweetForm(FlaskForm):
    content = TextAreaField('What\'s happening?', 
                           validators=[DataRequired(), Length(min=1, max=280)])
    submit = SubmitField('Tweet')

@tweet.route('/tweet/create', methods=['GET', 'POST'])
@login_required
def create():
    form = TweetForm()
    if form.validate_on_submit():
        # Create the tweet
        tweet_text = form.content.data
        new_tweet = Tweet(text=tweet_text, user_id=current_user.id)
        
        # Analyze sentiment
        sentiment_score, sentiment_label = analyze_sentiment(tweet_text)
        new_tweet.sentiment_score = sentiment_score
        new_tweet.sentiment_label = sentiment_label
        
        # Extract hashtags
        hashtags = extract_hashtags(tweet_text)
        
        # Add the tweet to the database
        db.session.add(new_tweet)
        db.session.commit()
        
        # Process and add hashtags
        for tag_text in hashtags:
            # Check if hashtag already exists in the database
            hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first()
            if not hashtag:
                # Create new hashtag if it doesn't exist
                hashtag = Hashtag(text=tag_text.lower())
                db.session.add(hashtag)
                db.session.commit()
            
            # Add hashtag to the tweet
            new_tweet.add_hashtag(hashtag)
        
        db.session.commit()
        
        flash('Your tweet has been posted!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('tweets/create.html', title='Create Tweet', form=form)

@tweet.route('/tweets/<int:tweet_id>/delete', methods=['POST'])
@login_required
def delete(tweet_id):
    tweet = Tweet.query.get_or_404(tweet_id)
    
    # Check if the current user is the author of the tweet
    if tweet.user_id != current_user.id:
        flash('You can only delete your own tweets!', 'danger')
        return redirect(url_for('dashboard.index'))
    
    db.session.delete(tweet)
    db.session.commit()
    
    flash('Your tweet has been deleted!', 'success')
    return redirect(url_for('dashboard.index'))

@tweet.route('/hashtag/<tag_text>')
@login_required
def hashtag(tag_text):
    # Find the hashtag in the database
    hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first_or_404()
    
    # Get all tweets with this hashtag
    tweets = hashtag.tweets.order_by(Tweet.timestamp.desc()).all()
    
    return render_template('tweets/hashtag.html', 
                          title=f'#{tag_text}',
                          hashtag=hashtag,
                          tweets=tweets)

# Utility function to extract hashtags from tweet text
def extract_hashtags(text):
    """Extract hashtags from tweet text."""
    # Find all words starting with # and containing word characters
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text)
    return hashtags

@tweet.route('/api/tweets', methods=['POST'])
@login_required
def api_create_tweet():
    """API endpoint for creating tweets via AJAX."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data.get('content'):
        return jsonify({'error': 'Tweet content is required'}), 400
    
    tweet_text = data['content']
    if len(tweet_text) > 280:
        return jsonify({'error': 'Tweet must be 280 characters or less'}), 400
    
    # Create the tweet
    new_tweet = Tweet(text=tweet_text, user_id=current_user.id)
    
    # Analyze sentiment
    sentiment_score, sentiment_label = analyze_sentiment(tweet_text)
    new_tweet.sentiment_score = sentiment_score
    new_tweet.sentiment_label = sentiment_label
    
    # Extract hashtags
    hashtags = extract_hashtags(tweet_text)
    
    # Add the tweet to the database
    db.session.add(new_tweet)
    db.session.commit()
    
    # Process and add hashtags
    for tag_text in hashtags:
        # Check if hashtag already exists in the database
        hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first()
        if not hashtag:
            # Create new hashtag if it doesn't exist
            hashtag = Hashtag(text=tag_text.lower())
            db.session.add(hashtag)
            db.session.commit()
        
        # Add hashtag to the tweet
        new_tweet.add_hashtag(hashtag)
    
    db.session.commit()
    
    # Return the newly created tweet info
    return jsonify({
        'id': new_tweet.id,
        'text': new_tweet.text,
        'timestamp': new_tweet.timestamp.isoformat(),
        'sentiment': new_tweet.sentiment_label,
        'author': current_user.username
    }), 201