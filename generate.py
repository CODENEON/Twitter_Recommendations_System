from app import create_app, db
from app.models import User, Tweet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import logging
import networkx as nx
from collections import defaultdict
import webbrowser
from flask import Flask, render_template, send_from_directory
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def view_clusters():
    return render_template('clusters.html')

@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/img', filename)

def generate_clusters():
    try:
        app_context = create_app()
        with app_context.app_context():
            logger.info("Starting cluster generation...")
            output_dir = 'static/img/'
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory created: {output_dir}")
            users = db.session.query(User)\
                .join(Tweet, User.id == Tweet.user_id)\
                .group_by(User.id)\
                .having(db.func.count(Tweet.id) > 0)\
                .limit(100)\
                .all()
            logger.info(f"Found {len(users)} users with tweets")
            if not users:
                logger.error("No users found with tweets")
                return
            user_data = []
            for user in users:
                tweets = db.session.query(Tweet)\
                    .filter(Tweet.user_id == user.id)\
                    .all()
                combined_text = " ".join([tweet.text for tweet in tweets])
                user_data.append({
                    'username': user.username,
                    'text': combined_text,
                    'id': user.id
                })
            logger.info("Created user data for clustering")
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([d['text'] for d in user_data])
            logger.info("Created TF-IDF matrix")
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(tfidf_matrix.toarray())
            logger.info("Performed PCA")
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            logger.info("Performed clustering")
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
            logger.info(f"Saved visualization to {plot_path}")
            G = nx.Graph()
            for i, user in enumerate(user_data):
                G.add_node(user['username'], 
                           cluster=clusters[i],
                           pos=(reduced_features[i, 0], reduced_features[i, 1]))
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            for i in range(len(user_data)):
                for j in range(i+1, len(user_data)):
                    if similarity_matrix[i, j] > 0.3:
                        G.add_edge(user_data[i]['username'], 
                                   user_data[j]['username'],
                                   weight=similarity_matrix[i, j])
            plt.figure(figsize=(12, 10))
            pos = nx.get_node_attributes(G, 'pos')
            clusters = [G.nodes[node]['cluster'] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, 
                                   node_color=clusters,
                                   cmap=plt.cm.viridis,
                                   node_size=100,
                                   alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            nx.draw_networkx_labels(G, pos, font_size=8)
            plt.title('User Network by Clusters')
            plt.axis('off')
            network_path = os.path.join(output_dir, 'network.png')
            plt.savefig(network_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved network visualization to {network_path}")
            print("\nCluster Information:")
            print(f"Total Users: {len(users)}")
            for cluster_id in range(3):
                cluster_users = [user_data[i]['username'] for i in range(len(user_data)) 
                                 if clusters[i] == cluster_id]
                print(f"\nCluster {cluster_id + 1}:")
                print(f"Size: {len(cluster_users)} users")
                print("Sample Users:")
                for user in cluster_users[:5]:
                    print(f"- {user}")
            logger.info("Cluster generation completed successfully")
    except Exception as e:
        logger.error(f"Error generating clusters: {str(e)}", exc_info=True)
    

def run_visualization_server():
    app.run(port=5001)

if __name__ == '__main__':
    generate_clusters()
    os.makedirs('templates', exist_ok=True)
    with open('templates/clusters.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>User Clusters Visualization</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container { margin-top: 2rem; }
        .card { margin-bottom: 1rem; }
        img { max-width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">User Network Clusters</h1>
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Network Visualization</h5>
                        <img src="/img/network.png" alt="User Network">
                        <p class="mt-3">
                            This network visualization shows how users are connected based on their tweet content similarity.
                            Nodes represent users, and edges represent content similarity between users.
                        </p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Cluster Visualization</h5>
                        <img src="/img/clusters.png" alt="User Clusters">
                        <p class="mt-3">
                            This visualization shows how users are grouped into clusters based on their tweet content.
                            Users in the same cluster tend to have similar tweeting patterns.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">About These Visualizations</h5>
                        <p class="card-text">
                            These visualizations provide two different views of how users are related:
                        </p>
                        <ul>
                            <li><strong>Network View:</strong> Shows direct connections between users based on content similarity</li>
                            <li><strong>Cluster View:</strong> Shows how users are grouped into distinct clusters</li>
                        </ul>
                        <p class="card-text">
                            <strong>How to interpret:</strong>
                            <ul>
                                <li>Nodes represent users</li>
                                <li>Edges represent content similarity</li>
                                <li>Colors represent different clusters</li>
                                <li>Closer nodes are more similar</li>
                            </ul>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        ''')
    server_thread = threading.Thread(target=run_visualization_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)
    webbrowser.open('http://localhost:5001')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
