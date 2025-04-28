import os
import nltk
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config


# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Flask extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    
    # Download NLTK data if not already present
    nltk_data_path = app.config['NLTK_DATA_PATH']
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)
    
    # Register blueprints
    from app.routes.auth import auth as auth_bp
    app.register_blueprint(auth_bp)
    
    from app.routes.dashboard import dashboard as dashboard_bp
    app.register_blueprint(dashboard_bp)
    
    from app.routes.tweet import tweet as tweet_bp
    app.register_blueprint(tweet_bp)
    
    from app.routes.user import user as user_bp
    app.register_blueprint(user_bp)
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    return app

from app import models