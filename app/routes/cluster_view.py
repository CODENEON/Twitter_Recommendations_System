from flask import Blueprint, render_template
from flask_login import login_required
from app import db
from app.models import User, Tweet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

cluster_view = Blueprint('cluster_view', __name__)

@cluster_view.route('/simple-clusters')
@login_required
def view_simple_clusters():
    try:
        output_dir = 'app/static/img/simple_clusters/'
        os.makedirs(output_dir, exist_ok=True)
        
        users = db.session.query(User)\
            .join(Tweet, User.id == Tweet.user_id)\
            .group_by(User.id)\
            .having(db.func.count(Tweet.id) > 0)\
            .limit(100)\
            .all()
        
        if not users:
            return render_template('cluster_view/error.html', 
                                 message="No users found with tweets")
        
        user_data = []
        for user in users:
            tweets = db.session.query(Tweet)\
                .filter(Tweet.user_id == user.id)\
                .all()
            
            combined_text = " ".join([tweet.text for tweet in tweets])
            user_data.append({
                'username': user.username,
                'text': combined_text
            })
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([d['text'] for d in user_data])
        
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=clusters, cmap='viridis', alpha=0.6)
        
        for i, user in enumerate(user_data):
            plt.annotate(user['username'], 
                        (reduced_features[i, 0], reduced_features[i, 1]),
                        fontsize=8)
        
        plt.title('User Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster')
        
        plot_path = os.path.join(output_dir, 'clusters.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        cluster_info = []
        for cluster_id in range(3):
            cluster_users = [user_data[i]['username'] for i in range(len(user_data)) 
                           if clusters[i] == cluster_id]
            cluster_info.append({
                'id': cluster_id,
                'size': len(cluster_users),
                'users': cluster_users[:5]
            })
        
        return render_template('cluster_view/simple.html',
                             plot_path='img/simple_clusters/clusters.png',
                             cluster_info=cluster_info,
                             total_users=len(users))
                             
    except Exception as e:
        return render_template('cluster_view/error.html', 
                             message=f"Error generating clusters: {str(e)}") 