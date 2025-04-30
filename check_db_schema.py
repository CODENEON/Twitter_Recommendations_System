import sqlite3
from app import create_app, db
from app.models import User, Tweet

def check_database_schema():
    # Connect to SQLite database
    conn = sqlite3.connect('twitter_analytics.db')
    cursor = conn.cursor()
    
    print("=== Database Schema ===")
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  Column: {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  Row count: {count}")
        
        # Get sample data
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
            sample = cursor.fetchone()
            print(f"  Sample row: {sample}")
    
    conn.close()
    
    # Now check using SQLAlchemy models
    print("\n=== SQLAlchemy Model Verification ===")
    app = create_app()
    with app.app_context():
        try:
            # Check User model
            users = User.query.all()
            print(f"\nUsers table:")
            print(f"Total users: {len(users)}")
            if users:
                user = users[0]
                print(f"Sample user: {user.username}")
                print(f"User ID: {user.id}")
                print(f"Tweets count: {user.tweets.count()}")
            
            # Check Tweet model
            tweets = Tweet.query.all()
            print(f"\nTweets table:")
            print(f"Total tweets: {len(tweets)}")
            if tweets:
                tweet = tweets[0]
                print(f"Sample tweet: {tweet.text[:100]}...")
                print(f"Tweet ID: {tweet.id}")
                print(f"User ID: {tweet.user_id}")
            
        except Exception as e:
            print(f"Error checking models: {str(e)}")

if __name__ == '__main__':
    check_database_schema() 