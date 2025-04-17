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

def cluster_users_by_tweets(n_clusters=5, method='kmeans', output_dir='app/static/img/clusters/'):
    """
    Cluster users based on their tweet content and generate visualizations.
    
    Args:
        n_clusters: Number of clusters to form (for KMeans)
        method: Clustering method ('kmeans' or 'dbscan')
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with cluster information and path to visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all users who have at least one tweet
    users = User.query.join(Tweet).group_by(User.id).having(db.func.count(Tweet.id) > 0).all()
    
    # Validate input parameters
    if n_clusters < 1:
        return {'error': 'Number of clusters must be at least 1'}
    
    if method == 'kmeans' and n_clusters > len(users):
        return {
            'error': f'Not enough users with tweets. Found {len(users)} users, but requested {n_clusters} clusters.'
        }
    
    # Collect user tweets
    user_data = []
    for user in users:
        tweets = user.tweets.all()
        # Combine all tweets for each user
        combined_text = " ".join([tweet.text for tweet in tweets])
        
        # Get sentiment scores and labels
        sentiment_scores = [tweet.sentiment_score for tweet in tweets]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Calculate sentiment distribution
        sentiment_distribution = {
            'positive': sum(1 for t in tweets if t.sentiment_label == 'positive'),
            'neutral': sum(1 for t in tweets if t.sentiment_label == 'neutral'),
            'negative': sum(1 for t in tweets if t.sentiment_label == 'negative')
        }
        
        # Get hashtags
        hashtags = []
        for tweet in tweets:
            for hashtag in tweet.hashtags:
                hashtags.append(hashtag.text)
        hashtag_text = " ".join(hashtags)
        
        # Extract common keywords
        word_counts = defaultdict(int)
        for tweet in tweets:
            # Simple tokenization - split by spaces and filter out common words
            words = tweet.text.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():  # Basic filtering
                    word_counts[word] += 1
        
        # Get top keywords
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        keyword_text = " ".join([word for word, count in top_keywords])
        
        user_data.append({
            'user_id': user.id,
            'username': user.username,
            'tweet_text': combined_text,
            'hashtag_text': hashtag_text,
            'keyword_text': keyword_text,
            'avg_sentiment': avg_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'tweet_count': len(tweets)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(user_data)
    
    # Feature extraction: TF-IDF on tweet text
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=1  # More permissive for small datasets
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tweet_text'])
    
    # Feature extraction: TF-IDF on hashtags
    if df['hashtag_text'].str.strip().str.len().sum() > 0:
        hashtag_vectorizer = TfidfVectorizer(
            max_features=50,
            min_df=1
        )
        hashtag_matrix = hashtag_vectorizer.fit_transform(df['hashtag_text'])
        hashtag_features = hashtag_matrix.toarray()
    else:
        # Create dummy hashtag features if none exist
        hashtag_features = np.zeros((len(df), 1))
    
    # Feature extraction: TF-IDF on keywords
    keyword_vectorizer = TfidfVectorizer(
        max_features=50,
        min_df=1
    )
    keyword_matrix = keyword_vectorizer.fit_transform(df['keyword_text'])
    
    # Create sentiment features
    sentiment_features = np.array([
        [
            row['avg_sentiment'], 
            row['sentiment_distribution']['positive'] / max(row['tweet_count'], 1), 
            row['sentiment_distribution']['negative'] / max(row['tweet_count'], 1),
            row['sentiment_distribution']['neutral'] / max(row['tweet_count'], 1)
        ] 
        for _, row in df.iterrows()
    ])
    
    # Combine features: TF-IDF, sentiment, tweet count
    # Convert sparse matrices to dense for concatenation
    feature_matrix = np.hstack([
        tfidf_matrix.toarray(),      # Tweet content features (60% weight)
        hashtag_features,            # Hashtag features (15% weight)
        keyword_matrix.toarray(),    # Keyword features (15% weight)
        sentiment_features,          # Sentiment features (10% weight)
        df[['tweet_count']].values   # Activity level feature
    ])
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    # Store the PCA coordinates
    df['x'] = reduced_features[:, 0]
    df['y'] = reduced_features[:, 1]
    
    # Clustering
    if method == 'kmeans':
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score only if n_clusters > 1 and we have enough data
        if n_clusters > 1 and len(df) >= n_clusters + 1:
            try:
                silhouette_avg = silhouette_score(scaled_features, df['cluster'])
                clustering_quality = f"Silhouette Score: {silhouette_avg:.3f}"
            except Exception as e:
                print(f"Error calculating silhouette score: {str(e)}")
                clustering_quality = "Silhouette score unavailable"
        else:
            clustering_quality = "Single cluster or insufficient data for silhouette score"
    
    elif method == 'dbscan':
        # Apply DBSCAN clustering
        eps = 0.5
        min_samples = min(3, max(2, len(df) // 4))  # Adaptive min_samples
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = dbscan.fit_predict(scaled_features)
        
        # Count number of clusters (excluding noise with label -1)
        n_clusters_found = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
        clustering_quality = f"DBSCAN found {n_clusters_found} clusters"
    
    # Generate multiple visualizations
    visualization_paths = {}
    
    # 1. Basic cluster visualization
    plt.figure(figsize=(12, 10))
    
    # Create a colormap
    colors = cm.nipy_spectral(np.linspace(0, 1, len(set(df['cluster']))))
    
    # Plot each cluster
    for i, color in zip(sorted(set(df['cluster'])), colors):
        cluster_label = f"Cluster {i}" if i >= 0 else "Noise"
        cluster_data = df[df['cluster'] == i]
        
        plt.scatter(
            cluster_data['x'], 
            cluster_data['y'],
            s=100, 
            color=color,
            label=f"{cluster_label} ({len(cluster_data)} users)",
            alpha=0.7
        )
    
    # Annotate points with usernames
    for i, row in df.iterrows():
        plt.annotate(
            row['username'],
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.8
        )
    
    plt.title(f'User Clustering Based on Tweet Content\n{clustering_quality}')
    plt.xlabel(f'Principal Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'user_clusters_{method}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    visualization_paths['cluster_plot'] = filepath
    
    # 2. Sentiment-enhanced visualization
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap for sentiment
    sentiment_norm = plt.Normalize(-1, 1)
    sentiment_cmap = plt.cm.RdYlGn  # Red for negative, yellow for neutral, green for positive
    
    # Plot each user
    scatter = plt.scatter(
        df['x'], 
        df['y'],
        s=df['tweet_count'] * 20 + 50,  # Size based on tweet count
        c=df['avg_sentiment'],  # Color based on sentiment
        cmap=sentiment_cmap,
        norm=sentiment_norm,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Sentiment Score')
    
    # Add cluster boundaries or centers
    for i in sorted(set(df['cluster'])):
        cluster_data = df[df['cluster'] == i]
        center_x = cluster_data['x'].mean()
        center_y = cluster_data['y'].mean()
        plt.scatter(
            center_x, 
            center_y,
            s=200, 
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9,
            label=f"Cluster {i} Center"
        )
    
    # Annotate points with usernames
    for i, row in df.iterrows():
        plt.annotate(
            row['username'],
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.8
        )
    
    plt.title('User Clustering with Sentiment Analysis\nSize = Tweet Count, Color = Sentiment')
    plt.xlabel(f'Principal Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the visualization
    sentiment_filename = f'user_clusters_sentiment_{timestamp}.png'
    sentiment_filepath = os.path.join(output_dir, sentiment_filename)
    plt.savefig(sentiment_filepath)
    plt.close()
    visualization_paths['sentiment_plot'] = sentiment_filepath
    
    # 3. Cluster characteristics visualization
    # Analyze clusters
    cluster_analysis = {}
    
    # For each cluster, create a visualization
    for cluster_id in sorted(set(df['cluster'])):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Skip if the cluster is empty or noise (-1 in DBSCAN)
        if len(cluster_data) == 0 or (method == 'dbscan' and cluster_id == -1):
            continue
        
        # Get combined tweet text for the cluster
        combined_text = " ".join(cluster_data['tweet_text'])
        
        # Extract top keywords using TF-IDF
        cluster_vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        try:
            cluster_tfidf = cluster_vectorizer.fit_transform([combined_text])
            feature_names = cluster_vectorizer.get_feature_names_out()
            
            # Get feature importance
            importance = np.squeeze(cluster_tfidf.toarray())
            
            # Create list of (word, importance) tuples
            top_keywords = [(feature_names[i], importance[i]) for i in importance.argsort()[-20:][::-1]]
        except Exception as e:
            print(f"Error extracting keywords for cluster {cluster_id}: {str(e)}")
            top_keywords = []
        
        # Calculate average sentiment
        avg_sentiment = cluster_data['avg_sentiment'].mean()
        
        # Get sentiment distribution
        sentiment_counts = {
            'positive': sum(row['sentiment_distribution']['positive'] for _, row in cluster_data.iterrows()),
            'neutral': sum(row['sentiment_distribution']['neutral'] for _, row in cluster_data.iterrows()),
            'negative': sum(row['sentiment_distribution']['negative'] for _, row in cluster_data.iterrows())
        }
        sentiment_total = sum(sentiment_counts.values())
        sentiment_distribution = {k: v/sentiment_total for k, v in sentiment_counts.items()} if sentiment_total > 0 else {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # Get most common hashtags
        hashtags = []
        for hashtag_text in cluster_data['hashtag_text']:
            hashtags.extend(hashtag_text.split())
        
        hashtag_counter = defaultdict(int)
        for hashtag in hashtags:
            hashtag_counter[hashtag] += 1
        
        top_hashtags = sorted(hashtag_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store analysis
        cluster_analysis[cluster_id] = {
            'size': len(cluster_data),
            'usernames': cluster_data['username'].tolist(),
            'avg_sentiment': float(avg_sentiment),  # Convert to float for JSON
            'sentiment_distribution': sentiment_distribution,
            'top_keywords': [(word, float(importance)) for word, importance in top_keywords],
            'top_hashtags': [(tag, count) for tag, count in top_hashtags],
            'user_ids': cluster_data['user_id'].tolist(),
            'tweet_count_total': int(cluster_data['tweet_count'].sum()),
            'tweet_count_avg': float(cluster_data['tweet_count'].mean())
        }
        
        # Create a visualization for this cluster
        plt.figure(figsize=(15, 10))
        
        # Create a 2x2 grid for different visualizations
        gs = plt.GridSpec(2, 2, figure=plt.gcf())
        
        # 1. Word cloud of top keywords
        ax1 = plt.subplot(gs[0, 0])
        if combined_text.strip():
            try:
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate(combined_text)
                
                ax1.imshow(wordcloud, interpolation='bilinear')
                ax1.set_title(f'Word Cloud for Cluster {cluster_id}')
                ax1.axis('off')
            except Exception as e:
                print(f"Error generating word cloud: {str(e)}")
                ax1.text(0.5, 0.5, "Word cloud generation failed", 
                        ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, "No text available for word cloud", 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Sentiment distribution pie chart
        ax2 = plt.subplot(gs[0, 1])
        sentiment_values = [sentiment_distribution['positive'], sentiment_distribution['neutral'], sentiment_distribution['negative']]
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_colors = ['#28a745', '#6c757d', '#dc3545']  # Green, Grey, Red
        
        ax2.pie(sentiment_values, labels=sentiment_labels, colors=sentiment_colors,
               autopct='%1.1f%%', startangle=90, wedgeprops={'alpha': 0.7})
        ax2.set_title(f'Sentiment Distribution (Avg: {avg_sentiment:.2f})')
        
        # 3. Top hashtags bar chart
        ax3 = plt.subplot(gs[1, 0])
        if top_hashtags:
            hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
            hashtag_df = hashtag_df.sort_values('Count').tail(10)  # Top 10
            
            sns.barplot(x='Count', y='Hashtag', data=hashtag_df, ax=ax3)
            ax3.set_title('Top Hashtags')
            ax3.set_xlabel('Count')
            ax3.set_ylabel('Hashtag')
        else:
            ax3.text(0.5, 0.5, "No hashtags found", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. User activity
        ax4 = plt.subplot(gs[1, 1])
        if not cluster_data.empty:
            user_activity = cluster_data[['username', 'tweet_count']].sort_values('tweet_count', ascending=False)
            
            if len(user_activity) > 10:
                user_activity = user_activity.head(10)
                
            sns.barplot(x='tweet_count', y='username', data=user_activity, ax=ax4)
            ax4.set_title('Most Active Users')
            ax4.set_xlabel('Tweet Count')
            ax4.set_ylabel('Username')
        else:
            ax4.text(0.5, 0.5, "No user activity data", 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.suptitle(f'Cluster {cluster_id} Analysis ({len(cluster_data)} Users)', fontsize=16)
        plt.tight_layout()
        
        # Save the cluster analysis visualization
        cluster_filename = f'cluster_{cluster_id}_analysis_{timestamp}.png'
        cluster_filepath = os.path.join(output_dir, cluster_filename)
        plt.savefig(cluster_filepath)
        plt.close()
        
        # Add to visualization paths
        visualization_paths[f'cluster_{cluster_id}_analysis'] = cluster_filepath
    
    # 4. Overall sentiment distribution
    plt.figure(figsize=(10, 8))
    
    # Create sentiment data
    sentiment_data = pd.DataFrame({
        'Cluster': [f'Cluster {cluster_id}' for cluster_id in cluster_analysis.keys()],
        'Positive': [cluster_analysis[cluster_id]['sentiment_distribution']['positive'] for cluster_id in cluster_analysis.keys()],
        'Neutral': [cluster_analysis[cluster_id]['sentiment_distribution']['neutral'] for cluster_id in cluster_analysis.keys()],
        'Negative': [cluster_analysis[cluster_id]['sentiment_distribution']['negative'] for cluster_id in cluster_analysis.keys()],
        'Size': [cluster_analysis[cluster_id]['size'] for cluster_id in cluster_analysis.keys()]
    })
    
    # Create stacked bar chart
    ax = sentiment_data.plot(
        x='Cluster',
        y=['Positive', 'Neutral', 'Negative'],
        kind='bar',
        stacked=True,
        color=['#28a745', '#6c757d', '#dc3545'],
        figsize=(10, 6)
    )
    
    # Add size as text on bars
    for i, size in enumerate(sentiment_data['Size']):
        ax.text(i, 1.05, f'Size: {size}', ha='center')
    
    plt.title('Sentiment Distribution Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    
    # Save the visualization
    sentiment_dist_filename = f'sentiment_distribution_{timestamp}.png'
    sentiment_dist_filepath = os.path.join(output_dir, sentiment_dist_filename)
    plt.savefig(sentiment_dist_filepath)
    plt.close()
    visualization_paths['sentiment_distribution'] = sentiment_dist_filepath
    
    return {
        'method': method,
        'n_clusters': n_clusters if method == 'kmeans' else len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0),
        'quality_metric': clustering_quality,
        'visualization_paths': visualization_paths,
        'cluster_analysis': cluster_analysis,
        'user_count': len(users),
        'raw_data': df.to_dict(orient='records'),  # Include raw data for further analysis
        'pca_variance_explained': pca.explained_variance_ratio_.tolist()
    }