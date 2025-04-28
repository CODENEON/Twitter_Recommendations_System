"""
Analysis API for backend use.
This module provides functions to run user clustering and other analytics 
without requiring frontend integration.
"""

import json
import pandas as pd
from datetime import datetime
import os
from app.services.user_clustering import cluster_users_by_tweets

def run_user_clustering_analysis(n_clusters=5, method='kmeans', output_dir='app/data/analysis/'):
    """
    Run the user clustering analysis and save results to disk.
    
    Args:
        n_clusters: Number of clusters to form (for KMeans)
        method: Clustering method ('kmeans' or 'dbscan')
        output_dir: Directory to save analysis results
        
    Returns:
        Path to the JSON results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run clustering analysis
    clustering_results = cluster_users_by_tweets(n_clusters=n_clusters, method=method)
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'error' in clustering_results:
        print(f"Clustering error: {clustering_results['error']}")
        return {'error': clustering_results['error']}
    
    # Prepare results for saving to JSON
    # We need to convert some data types that aren't JSON serializable
    serializable_results = {
        'method': clustering_results['method'],
        'n_clusters': clustering_results['n_clusters'],
        'quality_metric': clustering_results['quality_metric'],
        'visualization_path': clustering_results['visualization_path'],
        'user_count': clustering_results['user_count'],
        'timestamp': timestamp,
        'cluster_analysis': {}
    }
    
    # Convert cluster analysis data
    for cluster_id, data in clustering_results['cluster_analysis'].items():
        serializable_results['cluster_analysis'][str(cluster_id)] = {
            'size': data['size'],
            'usernames': data['usernames'],
            'avg_sentiment': float(data['avg_sentiment']),  # Ensure float type for JSON
            'top_keywords': data['top_keywords'],
            'top_hashtags': data['top_hashtags'],
            'user_ids': data['user_ids']
        }
    
    # Save results to JSON file
    results_filename = f'user_clustering_{method}_{timestamp}.json'
    results_filepath = os.path.join(output_dir, results_filename)
    
    with open(results_filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # Also save the raw data as CSV for potential further analysis
    if 'raw_data' in clustering_results:
        raw_data_df = pd.DataFrame(clustering_results['raw_data'])
        csv_filepath = os.path.join(output_dir, f'user_clustering_data_{timestamp}.csv')
        raw_data_df.to_csv(csv_filepath, index=False)
    
    return {
        'results_filepath': results_filepath,
        'visualization_path': clustering_results['visualization_path'],
        'csv_filepath': csv_filepath if 'raw_data' in clustering_results else None,
        'n_clusters': clustering_results['n_clusters'],
        'cluster_summary': {
            str(cluster_id): {
                'size': data['size'],
                'avg_sentiment': float(data['avg_sentiment']),
                'top_keywords': data['top_keywords'][:3] if data['top_keywords'] else [],
                'top_hashtags': data['top_hashtags'][:3] if data['top_hashtags'] else []
            }
            for cluster_id, data in clustering_results['cluster_analysis'].items()
        }
    }

def get_sentiment_by_cluster(clustering_results_filepath):
    """
    Extract sentiment analysis data grouped by cluster.
    
    Args:
        clustering_results_filepath: Path to the JSON clustering results file
        
    Returns:
        Dictionary with sentiment data by cluster
    """
    # Load clustering results
    with open(clustering_results_filepath, 'r') as f:
        clustering_results = json.load(f)
    
    sentiment_data = {}
    
    # Extract sentiment data by cluster
    for cluster_id, data in clustering_results['cluster_analysis'].items():
        sentiment_data[cluster_id] = {
            'avg_sentiment': data['avg_sentiment'],
            'size': data['size'],
            'label': 'positive' if data['avg_sentiment'] > 0.1 else 
                    'negative' if data['avg_sentiment'] < -0.1 else 'neutral'
        }
    
    return sentiment_data

def get_keyword_usage_by_cluster(clustering_results_filepath):
    """
    Extract keyword usage data grouped by cluster.
    
    Args:
        clustering_results_filepath: Path to the JSON clustering results file
        
    Returns:
        Dictionary with keyword data by cluster
    """
    # Load clustering results
    with open(clustering_results_filepath, 'r') as f:
        clustering_results = json.load(f)
    
    keyword_data = {}
    
    # Extract keyword data by cluster
    for cluster_id, data in clustering_results['cluster_analysis'].items():
        keyword_data[cluster_id] = {
            'top_keywords': data['top_keywords'],
            'size': data['size']
        }
    
    return keyword_data

def get_hashtag_usage_by_cluster(clustering_results_filepath):
    """
    Extract hashtag usage data grouped by cluster.
    
    Args:
        clustering_results_filepath: Path to the JSON clustering results file
        
    Returns:
        Dictionary with hashtag data by cluster
    """
    # Load clustering results
    with open(clustering_results_filepath, 'r') as f:
        clustering_results = json.load(f)
    
    hashtag_data = {}
    
    # Extract hashtag data by cluster
    for cluster_id, data in clustering_results['cluster_analysis'].items():
        hashtag_data[cluster_id] = {
            'top_hashtags': data['top_hashtags'],
            'size': data['size']
        }
    
    return hashtag_data

def get_all_cluster_analyses():
    """
    Get a list of all available cluster analyses.
    
    Returns:
        List of dictionaries with analysis metadata
    """
    analyses_dir = 'app/data/analysis/'
    
    # Create directory if it doesn't exist
    if not os.path.exists(analyses_dir):
        os.makedirs(analyses_dir)
    
    analyses = []
    
    # Find all JSON files in the analyses directory
    for filename in os.listdir(analyses_dir):
        if filename.startswith('user_clustering_') and filename.endswith('.json'):
            filepath = os.path.join(analyses_dir, filename)
            
            # Load the analysis file
            with open(filepath, 'r') as f:
                analysis_data = json.load(f)
            
            # Extract method and timestamp from filename
            parts = filename.split('_')
            method = parts[2]
            timestamp_str = parts[3].split('.')[0]
            
            # Format timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            analyses.append({
                'filepath': filepath,
                'method': method,
                'timestamp': formatted_timestamp,
                'n_clusters': analysis_data['n_clusters'],
                'user_count': analysis_data['user_count']
            })
    
    # Sort by timestamp, most recent first
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return analyses