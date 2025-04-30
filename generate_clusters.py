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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_clusters():
    try:
        # Create Flask app context
        app = create_app()
        with app.app_context():
            logger.info("Starting cluster generation...")
            
            # Create output directory
            output_dir = 'app/static/img/simple_clusters/'
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory created: {output_dir}")
            
            # Get users with tweets
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
            
            # Prepare data
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
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([d['text'] for d in user_data])
            logger.info("Created TF-IDF matrix")
            
            # Perform PCA
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(tfidf_matrix.toarray())
            logger.info("Performed PCA")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            logger.info("Performed clustering")
            
            # Generate visualization
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                c=clusters, cmap='viridis', alpha=0.6)
            
            # Add user labels
            for i, user in enumerate(user_data):
                plt.annotate(user['username'], 
                            (reduced_features[i, 0], reduced_features[i, 1]),
                            fontsize=8)
            
            plt.title('User Clusters')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.colorbar(scatter, label='Cluster')
            
            # Save the plot
            plot_path = os.path.join(output_dir, 'clusters.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved visualization to {plot_path}")
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes with cluster information
            for i, user in enumerate(user_data):
                G.add_node(user['username'], 
                          cluster=clusters[i],
                          pos=(reduced_features[i, 0], reduced_features[i, 1]))
            
            # Add edges based on similarity
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            for i in range(len(user_data)):
                for j in range(i+1, len(user_data)):
                    if similarity_matrix[i, j] > 0.3:  # Threshold for edge creation
                        G.add_edge(user_data[i]['username'], 
                                 user_data[j]['username'],
                                 weight=similarity_matrix[i, j])
            
            # Save network visualization
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
            
            # Print cluster information
            print("\nCluster Information:")
            print(f"Total Users: {len(users)}")
            
            for cluster_id in range(3):
                cluster_users = [user_data[i]['username'] for i in range(len(user_data)) 
                               if clusters[i] == cluster_id]
                print(f"\nCluster {cluster_id + 1}:")
                print(f"Size: {len(cluster_users)} users")
                print("Sample Users:")
                for user in cluster_users[:5]:  # Show first 5 users
                    print(f"- {user}")
            
            logger.info("Cluster generation completed successfully")
            
    except Exception as e:
        logger.error(f"Error generating clusters: {str(e)}", exc_info=True)

if __name__ == '__main__':
    generate_clusters() 