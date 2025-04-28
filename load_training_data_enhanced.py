import csv
from datetime import datetime
import re
from app import create_app, db
from app.models import User, Tweet, Hashtag

def clean_text(text):
    """Clean and normalize text data."""
    if not text:
        return ""
    # Remove extra quotes and whitespace
    text = text.strip().strip('"')
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_hashtags(text):
    """Extract hashtags from text."""
    if not text:
        return []
    return [tag.lower() for tag in re.findall(r'#(\w+)', text)]

def parse_timestamp(timestamp_str):
    """Parse Twitter timestamp string to datetime object."""
    try:
        # Remove timezone info and parse
        timestamp_str = timestamp_str.split(' PDT ')[0] + ' ' + timestamp_str.split(' PDT ')[1]
        return datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
    except (ValueError, IndexError):
        return datetime.utcnow()  # Return current time if parsing fails

def load_training_data():
    # Initialize Flask app
    app = create_app()
    
    total_processed = 0
    total_errors = 0
    batch_size = 1000
    
    with app.app_context():
        try:
            with open('training_data.csv', 'r', encoding='latin1') as file:
                # Use csv reader to handle quoted fields properly
                reader = csv.reader(file)
                
                # Skip header if exists
                next(reader, None)
                
                for row in reader:
                    try:
                        if len(row) < 6:
                            print(f"Skipping row with insufficient columns: {row}")
                            total_errors += 1
                            continue
                        
                        # Extract and clean data
                        sentiment = row[0].strip()
                        tweet_id = row[1].strip()
                        timestamp_str = row[2].strip()
                        username = clean_text(row[-2])  # Second to last column
                        text = clean_text(row[-1])      # Last column
                        
                        # Validate required fields
                        if not username or not text:
                            print(f"Skipping row with missing username or text: {row}")
                            total_errors += 1
                            continue
                        
                        # Parse sentiment (handle missing/invalid values)
                        try:
                            sentiment_score = float(sentiment)
                        except (ValueError, TypeError):
                            sentiment_score = 0.0  # Default to neutral
                        
                        # Determine sentiment label
                        if sentiment_score > 0:
                            sentiment_label = 'positive'
                        elif sentiment_score < 0:
                            sentiment_label = 'negative'
                        else:
                            sentiment_label = 'neutral'
                        
                        # Parse timestamp (handle missing/invalid values)
                        try:
                            timestamp = parse_timestamp(timestamp_str)
                        except (ValueError, IndexError):
                            timestamp = datetime.utcnow()  # Use current time as fallback
                        
                        # Create or get user
                        user = User.query.filter_by(username=username).first()
                        if not user:
                            user = User(
                                username=username,
                                email=f"{username}@example.com",  # Placeholder email
                                bio="Training data user",
                                member_since=timestamp
                            )
                            user.set_password('password')  # Placeholder password
                            db.session.add(user)
                            db.session.flush()  # Get the user ID
                        
                        # Create tweet
                        tweet = Tweet(
                            text=text,
                            timestamp=timestamp,
                            user_id=user.id,
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label
                        )
                        db.session.add(tweet)
                        db.session.flush()  # Get the tweet ID
                        
                        # Extract and process hashtags
                        hashtags = extract_hashtags(text)
                        for hashtag_text in hashtags:
                            if not hashtag_text:  # Skip empty hashtags
                                continue
                            
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
                        print(f"Problematic row: {row}")
                        total_errors += 1
                        db.session.rollback()
                        continue
                
                # Commit any remaining records
                if total_processed % batch_size != 0:
                    db.session.commit()
        
        except Exception as e:
            print(f"Error opening or reading file: {e}")
            return
    
    print(f"\nData loading completed!")
    print(f"Total tweets processed: {total_processed}")
    print(f"Total errors encountered: {total_errors}")
    print(f"Success rate: {(total_processed / (total_processed + total_errors)) * 100:.2f}%")

if __name__ == '__main__':
    load_training_data() 