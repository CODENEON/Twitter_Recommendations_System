import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
from collections import defaultdict
import os
from datetime import datetime
from app import db
from app.models import User, Tweet
import seaborn as sns
from wordcloud import WordCloud
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed
import dask.array as da

def process_user_tweets_parallel(users, client):
    """
    Process user tweets in parallel using Dask
    """
    # Convert user data to Dask DataFrame
    user_data = []
    for user in users:
        tweets = user.tweets.all()
        combined_text = " ".join([tweet.text for tweet in tweets])
        sentiment_scores = [tweet.sentiment_score for tweet in tweets]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        sentiment_distribution = {
            'positive': sum(1 for t in tweets if t.sentiment_label == 'positive'),
            'neutral': sum(1 for t in tweets if t.sentiment_label == 'neutral'),
            'negative': sum(1 for t in tweets if t.sentiment_label == 'negative')
        }
        
        hashtags = []
        for tweet in tweets:
            for hashtag in tweet.hashtags:
                hashtags.append(hashtag.text)
        hashtag_text = " ".join(hashtags)
        
        user_data.append({
            'user_id': user.id,
            'username': user.username,
            'tweet_text': combined_text,
            'hashtag_text': hashtag_text,
            'avg_sentiment': avg_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'tweet_count': len(tweets)
        })
    
    # Create Dask DataFrame
    ddf = dd.from_pandas(pd.DataFrame(user_data), npartitions=4)
    
    # Parallel feature extraction
    @delayed
    def extract_features(partition):
        # TF-IDF on tweet text
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
        tfidf_matrix = tfidf_vectorizer.fit_transform(partition['tweet_text'])
        
        # TF-IDF on hashtags
        hashtag_vectorizer = TfidfVectorizer(max_features=50, min_df=1)
        hashtag_matrix = hashtag_vectorizer.fit_transform(partition['hashtag_text'])
        
        # Sentiment features
        sentiment_features = np.array([
            [
                row['avg_sentiment'],
                row['sentiment_distribution']['positive'] / max(row['tweet_count'], 1),
                row['sentiment_distribution']['negative'] / max(row['tweet_count'], 1),
                row['sentiment_distribution']['neutral'] / max(row['tweet_count'], 1)
            ]
            for _, row in partition.iterrows()
        ])
        
        return tfidf_matrix, hashtag_matrix, sentiment_features
    
    # Process partitions in parallel
    results = []
    for partition in ddf.partitions:
        results.append(extract_features(partition))
    
    # Compute results
    computed_results = client.compute(results)
    
    # Combine results
    tfidf_matrices = []
    hashtag_matrices = []
    sentiment_features_list = []
    
    for result in computed_results:
        tfidf_matrices.append(result[0])
        hashtag_matrices.append(result[1])
        sentiment_features_list.append(result[2])
    
    # Combine matrices
    final_tfidf = da.concatenate([da.from_array(m.toarray()) for m in tfidf_matrices])
    final_hashtag = da.concatenate([da.from_array(m.toarray()) for m in hashtag_matrices])
    final_sentiment = da.concatenate([da.from_array(f) for f in sentiment_features_list])
    
    # Combine all features
    feature_matrix = da.concatenate([
        final_tfidf,
        final_hashtag,
        final_sentiment,
        da.from_array(ddf[['tweet_count']].values)
    ], axis=1)
    
    return feature_matrix, ddf

def cluster_users_by_tweets(n_clusters=5, method='kmeans', output_dir='app/static/img/clusters/'):
    """
    Cluster users based on their tweet content using parallel processing
    """
    # Create Dask client
    client = Client()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all users who have at least one tweet
    users = User.query.join(Tweet).group_by(User.id).having(db.func.count(Tweet.id) > 0).all()
    
    # Process data in parallel
    feature_matrix, ddf = process_user_tweets_parallel(users, client)
    
    # Scale features in parallel
    scaler = StandardScaler()
    scaled_features = da.from_array(scaler.fit_transform(feature_matrix.compute()))
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features.compute())
    
    # Store PCA coordinates
    ddf['x'] = reduced_features[:, 0]
    ddf['y'] = reduced_features[:, 1]
    
    # Clustering
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        ddf['cluster'] = kmeans.fit_predict(scaled_features.compute())
    elif method == 'dbscan':
        eps = 0.5
        min_samples = min(3, max(2, len(ddf) // 4))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        ddf['cluster'] = dbscan.fit_predict(scaled_features.compute())
    
    # Generate visualizations in parallel
    @delayed
    def generate_visualization(cluster_data, cluster_id):
        plt.figure(figsize=(15, 10))
        
        # Scatter plot of PCA components
        plt.scatter(cluster_data['x'], cluster_data['y'], alpha=0.6)
        plt.title(f'Cluster {cluster_id} - User Distribution')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Add user labels
        for idx, row in cluster_data.iterrows():
            plt.annotate(row['username'], (row['x'], row['y']), fontsize=8)
        
        visualization_path = f"{output_dir}cluster_{cluster_id}.png"
        plt.savefig(visualization_path)
        plt.close()
        return visualization_path
    
    # Process clusters in parallel
    visualization_tasks = []
    for cluster_id in sorted(set(ddf['cluster'])):
        cluster_data = ddf[ddf['cluster'] == cluster_id]
        visualization_tasks.append(generate_visualization(cluster_data, cluster_id))
    
    # Compute visualizations
    visualization_paths = client.compute(visualization_tasks)
    
    return {
        'method': method,
        'n_clusters': n_clusters if method == 'kmeans' else len(set(ddf['cluster'])) - (1 if -1 in ddf['cluster'] else 0),
        'visualization_paths': visualization_paths,
        'user_count': len(users)
    }