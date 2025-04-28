import pandas as pd
from datetime import datetime
import re
from app import create_app, db
from app.models import User, Tweet, Hashtag, TweetHashtag

def parse_timestamp(timestamp_str):
    """Parse Twitter timestamp string to datetime object"""
    # Remove quotes if present
    timestamp_str = timestamp_str.strip('"')
    return datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S PDT %Y')

def extract_hashtags(text):
    """Extract hashtags from tweet text"""
    # Remove quotes if present
    text = text.strip('"')
    return re.findall(r'#(\w+)', text)

def clean_text(text):
    """Clean text by removing quotes and handling special characters"""
    return text.strip('"').replace('\\n', '\n').replace('\\t', '\t')

def load_training_data():
    # Initialize Flask app
    app = create_app()
    
    total_processed = 0
    batch_size = 1000
    
    with app.app_context():
        with open('training_data.csv', 'r', encoding='latin1') as file:
            # Skip header if exists
            next(file, None)
            
            for line in file:
                try:
                    # Split line by comma, but not within quotes
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    # Extract data
                    sentiment = parts[0]
                    username = parts[-2]  # Second to last column
                    text = parts[-1]      # Last column
                    
                    # Clean data
                    username = clean_text(username)
                    text = clean_text(text)
                    
                    # Create or get user
                    user = User.query.filter_by(username=username).first()
                    if not user:
                        user = User(
                            username=username,
                            email=f"{username}@example.com",  # Placeholder email
                            bio="Training data user"
                        )
                        user.set_password('password')  # Placeholder password
                        db.session.add(user)
                        db.session.flush()  # Get the user ID
                    
                    # Create tweet
                    tweet = Tweet(
                        id=total_processed + 1,  # Use incremental ID
                        text=text,
                        timestamp=datetime.utcnow(),  # Use current time as fallback
                        user_id=user.id,
                        sentiment_score=float(sentiment),
                        sentiment_label='positive' if float(sentiment) > 0 else 'negative' if float(sentiment) < 0 else 'neutral'
                    )
                    db.session.add(tweet)
                    
                    # Extract and process hashtags
                    hashtags = extract_hashtags(text)
                    for hashtag_text in hashtags:
                        hashtag = Hashtag.query.filter_by(text=hashtag_text.lower()).first()
                        if not hashtag:
                            hashtag = Hashtag(text=hashtag_text.lower())
                            db.session.add(hashtag)
                            db.session.flush()
                        
                        # Add hashtag to tweet
                        tweet.hashtags.append(hashtag)
                    
                    total_processed += 1
                    
                    # Commit in batches
                    if total_processed % batch_size == 0:
                        db.session.commit()
                        print(f"Processed {total_processed} tweets...")
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    db.session.rollback()
                    continue
            
            # Commit any remaining records
            if total_processed % batch_size != 0:
                db.session.commit()
    
    print(f"Data loading completed! Total tweets processed: {total_processed}")

if __name__ == '__main__':
    load_training_data() 