# Twitter Recommender System

A Flask-based web application that provides personalized recommendations for Twitter users based on their interests and interactions.

## Features

- User authentication and profile management
- Tweet creation and interaction (likes, retweets)
- Hashtag tracking and analysis
- Personalized recommendations based on:
  - User interests and hashtag usage
  - Similar users' content
  - Trending topics
- Social network analysis
- Hashtag relationship analysis

## Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite (development), PostgreSQL (production)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Authentication**: Flask-Login
- **Data Analysis**: scikit-learn, pandas, numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/twitter-recommender.git
cd twitter-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

6. Run the application:
```bash
flask run
```

## API Endpoints

- `/api/recommendations` - Get personalized recommendations
- `/api/network/analysis` - Analyze social network connections
- `/api/hashtags/analysis` - Analyze hashtag relationships and trends

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
