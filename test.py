from app import create_app
from app.services.user_clustering import cluster_users_by_tweets
import sys

app = create_app()

# Create an application context
with app.app_context():
    try:
        # Now you can run your clustering analysis
        results = cluster_users_by_tweets(n_clusters=3, method='kmeans')
        print("Clustering completed successfully!")
        
        # Check if we have the visualization paths dictionary
        if 'visualization_paths' in results:
            print("\nVisualizations created:")
            for viz_type, path in results['visualization_paths'].items():
                print(f"- {viz_type}: {path}")
            
            # If you want to highlight the main cluster plot
            if 'cluster_plot' in results['visualization_paths']:
                print(f"\nMain cluster visualization: {results['visualization_paths']['cluster_plot']}")
        
        # For backward compatibility, check for the old key too
        elif 'visualization_path' in results:
            print(f"Visualization saved to: {results['visualization_path']}")
        
        # Print cluster information
        print(f"\nFound {results['n_clusters']} clusters with {results['user_count']} users")
        print(f"Clustering quality: {results['quality_metric']}")
        
        # Print summary of each cluster
        print("\nCluster Summary:")
        for cluster_id, data in results['cluster_analysis'].items():
            print(f"Cluster {cluster_id}:")
            print(f"  Size: {data['size']} users")
            print(f"  Avg Sentiment: {data['avg_sentiment']:.2f}")
            print(f"  Top Keywords: {', '.join([k[0] for k in data['top_keywords'][:5]])}" if data['top_keywords'] else "  No significant keywords")
            print(f"  Top Hashtags: {', '.join([h[0] for h in data['top_hashtags'][:5]])}" if data['top_hashtags'] else "  No hashtags")
            print()
            
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback
        traceback.print_exc()