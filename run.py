from app import create_app, db
from app.models import User, Tweet, Hashtag, TweetHashtag, Follow

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db, 
        'User': User, 
        'Tweet': Tweet, 
        'Hashtag': Hashtag,
        'TweetHashtag': TweetHashtag,
        'Follow': Follow
    }

if __name__ == '__main__':
    app.run(debug=True)