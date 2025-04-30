from app import create_app, db
from app.models import User, Tweet

def check_database():
    app = create_app()
    with app.app_context():
        try:
            # Check total users
            total_users = User.query.count()
            print(f"Total users in database: {total_users}")
            
            # Check users with tweets
            users_with_tweets = User.query.join(Tweet).group_by(User.id).having(db.func.count(Tweet.id) > 0).all()
            print(f"Users with tweets: {len(users_with_tweets)}")
            
            if users_with_tweets:
                # Print sample data
                sample_user = users_with_tweets[0]
                print(f"\nSample user: {sample_user.username}")
                print(f"User ID: {sample_user.id}")
                print(f"Number of tweets: {len(sample_user.tweets.all())}")
                
                # Print first few tweets
                print("\nFirst 3 tweets:")
                for tweet in sample_user.tweets.limit(3).all():
                    print(f"- {tweet.text[:100]}...")
            else:
                print("\nNo users found with tweets!")
                
        except Exception as e:
            print(f"Error checking database: {str(e)}")

if __name__ == '__main__':
    check_database() 