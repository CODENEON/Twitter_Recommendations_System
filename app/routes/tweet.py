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
        tweet_text = form.content.data
        new_tweet = Tweet(text=tweet_text, user_id=current_user.id)
        
        sentiment_score, sentiment_label = analyze_sentiment(tweet_text)
        new_tweet.sentiment_score = sentiment_score
        new_tweet.sentiment_label = sentiment_label
        
        hashtags = extract_hashtags(tweet_text)
        
        db.session.add(new_tweet)
        db.session.commit()
        
        for tag_text in hashtags:
            hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first()
            if not hashtag:
                hashtag = Hashtag(text=tag_text.lower())
                db.session.add(hashtag)
                db.session.commit()
            
            new_tweet.add_hashtag(hashtag)
        
        db.session.commit()
        
        flash('Your tweet has been posted!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('tweets/create.html', title='Create Tweet', form=form)

@tweet.route('/tweets/<int:tweet_id>/delete', methods=['POST'])
@login_required
def delete(tweet_id):
    tweet = Tweet.query.get_or_404(tweet_id)
    
    if tweet.user_id != current_user.id:
        flash('You can only delete your own tweets!', 'danger')
        return redirect(url_for('dashboard.index'))
    
    db.session.delete(tweet)
    db.session.commit()
    
    flash('Your tweet has been deleted!', 'success')
    return redirect(url_for('dashboard.index'))

def get_related_hashtags(hashtag_text, limit=5):
    hashtag = Hashtag.query.filter_by(text=hashtag_text.lower()).first()
    if not hashtag:
        return []
    
    related_hashtags = db.session.query(
        Hashtag.text,
        db.func.count(Hashtag.id).label('count')
    ).join(
        Hashtag.tweets
    ).filter(
        Hashtag.id != hashtag.id,
        Hashtag.tweets.any(Tweet.id.in_([t.id for t in hashtag.tweets]))
    ).group_by(
        Hashtag.id
    ).order_by(
        db.desc('count')
    ).limit(limit).all()
    
    return [{'text': tag, 'count': count} for tag, count in related_hashtags]

@tweet.route('/hashtag/<tag_text>')
@login_required
def hashtag(tag_text):
    hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first_or_404()
    
    tweets = hashtag.tweets.order_by(Tweet.timestamp.desc()).all()
    
    related_hashtags = get_related_hashtags(tag_text)
    
    return render_template('tweets/hashtag.html', 
                          title=f'#{tag_text}',
                          hashtag=hashtag,
                          tweets=tweets,
                          related_hashtags=related_hashtags)

def extract_hashtags(text):
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text)
    return hashtags

@tweet.route('/api/tweets', methods=['POST'])
@login_required
def api_create_tweet():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data.get('content'):
        return jsonify({'error': 'Tweet content is required'}), 400
    
    tweet_text = data['content']
    if len(tweet_text) > 280:
        return jsonify({'error': 'Tweet must be 280 characters or less'}), 400
    
    new_tweet = Tweet(text=tweet_text, user_id=current_user.id)
    
    sentiment_score, sentiment_label = analyze_sentiment(tweet_text)
    new_tweet.sentiment_score = sentiment_score
    new_tweet.sentiment_label = sentiment_label
    
    hashtags = extract_hashtags(tweet_text)
    
    db.session.add(new_tweet)
    db.session.commit()
    
    for tag_text in hashtags:
        hashtag = Hashtag.query.filter_by(text=tag_text.lower()).first()
        if not hashtag:
            hashtag = Hashtag(text=tag_text.lower())
            db.session.add(hashtag)
            db.session.commit()
        
        new_tweet.add_hashtag(hashtag)
    
    db.session.commit()
    
    return jsonify({
        'id': new_tweet.id,
        'text': new_tweet.text,
        'timestamp': new_tweet.timestamp.isoformat(),
        'sentiment': new_tweet.sentiment_label,
        'author': current_user.username
    }), 201

@tweet.route('/api/related_hashtags/<tag_text>')
@login_required
def api_related_hashtags(tag_text):
    related_hashtags = get_related_hashtags(tag_text)
    return jsonify(related_hashtags)