from app import create_app, db
from app.models import User, Tweet
import sqlite3

def check_database_data():
    # First check using SQLite directly
    print("=== Direct SQLite Check ===")
    conn = sqlite3.connect('twitter_analytics.db')
    cursor = conn.cursor()
    
    # Check users with tweets
    cursor.execute("""
        SELECT u.id, u.username, COUNT(t.id) as tweet_count 
        FROM users u 
        JOIN tweets t ON u.id = t.user_id 
        GROUP BY u.id 
        HAVING COUNT(t.id) > 0 
        ORDER BY tweet_count DESC 
        LIMIT 5;
    """)
    users_with_tweets = cursor.fetchall()
    print("\nTop 5 users with tweets:")
    for user in users_with_tweets:
        print(f"User ID: {user[0]}, Username: {user[1]}, Tweet Count: {user[2]}")
    
    # Check total counts
    cursor.execute("SELECT COUNT(*) FROM users;")
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM tweets;")
    total_tweets = cursor.fetchone()[0]
    cursor.execute("""
        SELECT COUNT(DISTINCT user_id) 
        FROM tweets;
    """)
    users_with_tweets_count = cursor.fetchone()[0]
    
    print(f"\nTotal users: {total_users}")
    print(f"Total tweets: {total_tweets}")
    print(f"Users with tweets: {users_with_tweets_count}")
    
    conn.close()
    
    # Now check using SQLAlchemy
    print("\n=== SQLAlchemy Check ===")
    app = create_app()
    with app.app_context():
        try:
            # Get users with tweets
            users = db.session.query(User)\
                .join(Tweet, User.id == Tweet.user_id)\
                .group_by(User.id)\
                .having(db.func.count(Tweet.id) > 0)\
                .order_by(db.func.count(Tweet.id).desc())\
                .limit(5)\
                .all()
            
            print("\nTop 5 users with tweets (SQLAlchemy):")
            for user in users:
                tweet_count = db.session.query(Tweet)\
                    .filter(Tweet.user_id == user.id)\
                    .count()
                print(f"User ID: {user.id}, Username: {user.username}, Tweet Count: {tweet_count}")
                
                # Print sample tweet
                sample_tweet = db.session.query(Tweet)\
                    .filter(Tweet.user_id == user.id)\
                    .first()
                if sample_tweet:
                    print(f"  Sample tweet: {sample_tweet.text[:100]}...")
            
        except Exception as e:
            print(f"Error in SQLAlchemy check: {str(e)}")

if __name__ == '__main__':
    check_database_data() 