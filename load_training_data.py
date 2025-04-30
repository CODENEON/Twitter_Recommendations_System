import csv
from datetime import datetime
import re
from app import create_app, db
from app.models import User, Tweet, Hashtag

def clean_text(text):

    if not text:
        return ""
    
    text = text.strip().strip('"')
    
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_hashtags(text):
    
    if not text:
        return []
    return [tag.lower() for tag in re.findall(r'#(\w+)', text)]

def parse_timestamp(timestamp_str):
    
    try:
        
        timestamp_str = timestamp_str.split(' PDT ')[0] + ' ' + timestamp_str.split(' PDT ')[1]
        return datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
    except (ValueError, IndexError):
        return datetime.utcnow()  

def load_training_data():
    
    app = create_app()
    
    total_processed = 0
    total_errors = 0
    total_duplicates = 0
    batch_size = 1000
    
    with app.app_context():
        try:
            with open('training_data.csv', 'r', encoding='latin1') as file:
                
                reader = csv.reader(file)
                
                next(reader, None)
                
                for row in reader:
                    try:
                        if len(row) < 6:
                            print(f"Skipping row with insufficient columns: {row}")
                            total_errors += 1
                            continue
                        
                        
                        sentiment = row[0].strip()
                        tweet_id = row[1].strip()
                        timestamp_str = row[2].strip()
                        username = clean_text(row[-2])  
                        text = clean_text(row[-1])      
                        
                        
                        if not username or not text:
                            print(f"Skipping row with missing username or text: {row}")
                            total_errors += 1
                            continue
                        
                        
                        try:
                            sentiment_score = float(sentiment)
                        except (ValueError, TypeError):
                            sentiment_score = 0.0  
                        
                        
                        if sentiment_score > 0:
                            sentiment_label = 'positive'
                        elif sentiment_score < 0:
                            sentiment_label = 'negative'
                        else:
                            sentiment_label = 'neutral'
                        
                        
                        try:
                            timestamp = parse_timestamp(timestamp_str)
                        except (ValueError, IndexError):
                            timestamp = datetime.utcnow()  
                        
                        
                        user = User.query.filter_by(username=username).first()
                        if not user:
                            user = User(
                                username=username,
                                email=f"{username}@example.com",  
                                bio="Training data user",
                                member_since=timestamp
                            )
                            user.set_password('password')  
                            db.session.add(user)
                            db.session.flush()  
                        
                        
                        existing_tweet = Tweet.query.filter_by(
                            user_id=user.id,
                            text=text
                        ).first()
                        
                        if existing_tweet:
                            total_duplicates += 1
                            continue
                        
                        
                        tweet = Tweet(
                            text=text,
                            timestamp=timestamp,
                            user_id=user.id,
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label
                        )
                        db.session.add(tweet)
                        db.session.flush()  
                        
                        
                        hashtags = extract_hashtags(text)
                        for hashtag_text in hashtags:
                            if not hashtag_text:  
                                continue
                            
                            hashtag = Hashtag.query.filter_by(text=hashtag_text.lower()).first()
                            if not hashtag:
                                hashtag = Hashtag(text=hashtag_text.lower())
                                db.session.add(hashtag)
                                db.session.flush()
                            
                            
                            tweet.hashtags.append(hashtag)
                        
                        total_processed += 1
                        
                        
                        if total_processed % batch_size == 0:
                            db.session.commit()
                            print(f"Processed {total_processed} tweets...")
                    
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        print(f"Problematic row: {row}")
                        total_errors += 1
                        db.session.rollback()
                        continue
                
                
                if total_processed % batch_size != 0:
                    db.session.commit()
        
        except Exception as e:
            print(f"Error opening or reading file: {e}")
            return
    
    print(f"\nData loading completed!")
    print(f"Total tweets processed: {total_processed}")
    print(f"Total duplicates skipped: {total_duplicates}")
    print(f"Total errors encountered: {total_errors}")
    print(f"Success rate: {(total_processed / (total_processed + total_errors + total_duplicates)) * 100:.2f}%")

if __name__ == '__main__':
    load_training_data() 