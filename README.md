# Twitter Recommender System

A comprehensive Twitter-like application with advanced recommendation and sentiment analysis features.

## Features

### 1. User Management
- **User Authentication**
  - Secure registration and login system
  - Password hashing for security
  - Session management
- **User Profiles**
  - Customizable profile information
  - Profile picture upload
  - User bio and personal details
- **Social Features**
  - Follow/Unfollow other users
  - View user's tweets and activity
  - User activity timeline

### 2. Tweet Management
- **Tweet Creation**
  - Create tweets with 280 character limit
  - Real-time character count
  - Quick tweet creation from dashboard
- **Tweet Interactions**
  - Like/Unlike tweets
  - Delete own tweets
  - View tweet details
  - Reply to tweets
- **Tweet Organization**
  - Automatic hashtag extraction
  - Sentiment analysis
  - Tweet categorization

### 3. Hashtag System
- **Hashtag Features**
  - Automatic hashtag extraction from tweets
  - Hashtag pages showing related tweets
  - Related hashtags based on co-occurrence
  - Trending hashtags
- **Hashtag Discovery**
  - Hashtag search functionality
  - Popular hashtags
  - Related hashtag suggestions

### 4. Dashboard
- **Feed Management**
  - Recent tweets feed
  - Tweets from followed users
  - Personalized content
- **Recommendations**
  - Tweet recommendations
  - User recommendations
  - Hashtag recommendations
- **Analytics**
  - Sentiment statistics
  - User activity metrics
  - Trending content

### 5. Sentiment Analysis
- **Tweet Analysis**
  - Automatic sentiment analysis
  - Positive/Negative/Neutral classification
  - Sentiment score calculation
- **Visualization**
  - Sentiment distribution charts
  - User sentiment trends
  - Comparative sentiment analysis

### 6. Recommendation System
- **Content-Based Recommendations**
  - Tweet similarity matching
  - Hashtag-based recommendations
  - Keyword-based suggestions
- **Collaborative Filtering**
  - User similarity analysis
  - Network-based recommendations
  - Activity-based suggestions
- **Hybrid Recommendations**
  - Combined content and collaborative filtering
  - Personalized feed generation
  - Context-aware suggestions

### 7. User Clustering
- **Clustering Algorithms**
  - TF-IDF based user clustering
  - KMeans clustering
  - Community detection
- **Analysis Features**
  - User similarity analysis
  - Community visualization
  - Cluster statistics

### 8. API Endpoints
- **Tweet API**
  - Create tweets
  - Get tweet details
  - Delete tweets
- **Dashboard API**
  - Get feed data
  - Get recommendations
  - Get statistics
- **Hashtag API**
  - Get hashtag data
  - Get related hashtags
  - Get trending hashtags
- **User API**
  - Get user recommendations
  - Get user activity
  - Get user statistics

### 9. Data Visualization
- **Charts and Graphs**
  - Sentiment distribution
  - User clustering
  - Trending hashtags
  - Activity metrics
- **Interactive Visualizations**
  - Real-time updates
  - Interactive charts
  - Customizable views

### 10. Search Functionality
- **Search Features**
  - Hashtag search
  - User search
  - Tweet search
- **Advanced Search**
  - Filter by sentiment
  - Filter by date
  - Filter by popularity

## Technical Stack
- **Backend**: Python, Flask
- **Database**: SQLAlchemy, Neo4j
- **Machine Learning**: scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Matplotlib, Seaborn
- **Processing**: Dask for parallel processing

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Initialize the database
5. Run the application:
   ```bash
   python run.py
   ```

## Configuration
- Create a `.env` file with necessary configurations
- Set up database connections
- Configure API keys if needed
- Set up email service for notifications

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
