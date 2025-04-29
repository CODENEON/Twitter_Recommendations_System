"""
Analysis API for backend use with parallel processing.
This module provides functions to run user clustering and other analytics 
without requiring frontend integration, using Dask for parallel computation.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Dask imports for parallel processing
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask import delayed

# Import the clustering function
from app.services.user_clustering import cluster_users_by_tweets

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Dask client for reuse across functions
_dask_client = None

def get_dask_client(n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Get or create a Dask client for parallel processing.
    
    Args:
        n_workers: Number of worker processes (None = auto-detect)
        threads_per_worker: Number of threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Dask client instance
    """
    global _dask_client
    
    if _dask_client is None or not _dask_client.status == 'running':
        logger.info("Starting Dask client for parallel processing...")
        
        # Create a local cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        # Create a client
        _dask_client = Client(cluster)
        logger.info(f"Dask dashboard available at: {_dask_client.dashboard_link}")
    
    return _dask_client

def close_dask_client():
    """Close the Dask client when done."""
    global _dask_client
    
    if _dask_client is not None:
        logger.info("Closing Dask client...")
        _dask_client.close()
        _dask_client = None

def run_user_clustering_analysis(n_clusters=5, method='kmeans', output_dir='app/data/analysis/',
                                n_workers=None, threads_per_worker=2, memory_limit='2GB'):
    """
    Run the user clustering analysis in parallel and save results to disk.
    
    Args:
        n_clusters: Number of clusters to form (for KMeans)
        method: Clustering method ('kmeans' or 'dbscan')
        output_dir: Directory to save analysis results
        n_workers: Number of Dask workers (None = auto-detect)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Path to the JSON results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Dask client for parallel processing
    client = get_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    try:
        logger.info(f"Running user clustering with method={method}, n_clusters={n_clusters}")
        
        # Run clustering analysis (already uses Dask internally)
        clustering_results = cluster_users_by_tweets(n_clusters=n_clusters, method=method)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'error' in clustering_results:
            logger.error(f"Clustering error: {clustering_results['error']}")
            return {'error': clustering_results['error']}
        
        # Prepare results for JSON serialization in parallel
        @delayed
        def process_cluster_data(cluster_id, data):
            """Process a single cluster's data in parallel."""
            return str(cluster_id), {
                'size': data['size'],
                'usernames': data['usernames'],
                'avg_sentiment': float(data['avg_sentiment']),
                'top_keywords': data['top_keywords'],
                'top_hashtags': data['top_hashtags'],
                'user_ids': data['user_ids']
            }
        
        # Schedule parallel processing of cluster data
        cluster_tasks = []
        for cluster_id, data in clustering_results['cluster_analysis'].items():
            cluster_tasks.append(process_cluster_data(cluster_id, data))
        
        # Compute all cluster data in parallel
        processed_clusters = client.compute(cluster_tasks)
        cluster_analysis = dict(processed_clusters)
        
        # Build the serializable results object
        serializable_results = {
            'method': clustering_results['method'],
            'n_clusters': clustering_results['n_clusters'],
            'quality_metric': clustering_results['quality_metric'],
            'visualization_path': clustering_results['visualization_path'],
            'user_count': clustering_results['user_count'],
            'timestamp': timestamp,
            'cluster_analysis': cluster_analysis
        }
        
        # Save results to JSON file
        results_filename = f'user_clustering_{method}_{timestamp}.json'
        results_filepath = os.path.join(output_dir, results_filename)
        
        with open(results_filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # Process and save raw data if available
        if 'raw_data' in clustering_results:
            # Convert to Dask DataFrame for parallel processing
            raw_data_df = pd.DataFrame(clustering_results['raw_data'])
            dask_df = dd.from_pandas(raw_data_df, npartitions=min(10, len(raw_data_df) // 100 + 1))
            
            # Write to CSV in parallel
            csv_filepath = os.path.join(output_dir, f'user_clustering_data_{timestamp}.csv')
            logger.info(f"Saving raw data to {csv_filepath}...")
            dask_df.to_csv(csv_filepath, index=False, single_file=True)
        else:
            csv_filepath = None
        
        # Prepare the response (this is quick, no need to parallelize)
        cluster_summary = {}
        for cluster_id, data in clustering_results['cluster_analysis'].items():
            cluster_summary[str(cluster_id)] = {
                'size': data['size'],
                'avg_sentiment': float(data['avg_sentiment']),
                'top_keywords': data['top_keywords'][:3] if data['top_keywords'] else [],
                'top_hashtags': data['top_hashtags'][:3] if data['top_hashtags'] else []
            }
        
        logger.info(f"Clustering analysis complete. Results saved to {results_filepath}")
        
        return {
            'results_filepath': results_filepath,
            'visualization_path': clustering_results['visualization_path'],
            'csv_filepath': csv_filepath,
            'n_clusters': clustering_results['n_clusters'],
            'cluster_summary': cluster_summary
        }
    
    finally:
        # Don't close the client here to allow reuse across functions
        pass

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
    
    # Get Dask client for parallel processing
    client = get_dask_client()
    
    try:
        # Define a function to process each cluster
        @delayed
        def process_cluster_sentiment(cluster_id, data):
            """Process sentiment data for a single cluster."""
            return cluster_id, {
                'avg_sentiment': data['avg_sentiment'],
                'size': data['size'],
                'label': 'positive' if data['avg_sentiment'] > 0.1 else 
                         'negative' if data['avg_sentiment'] < -0.1 else 'neutral'
            }
        
        # Schedule parallel processing of cluster sentiment
        sentiment_tasks = []
        for cluster_id, data in clustering_results['cluster_analysis'].items():
            sentiment_tasks.append(process_cluster_sentiment(cluster_id, data))
        
        # Compute all sentiment data in parallel
        sentiment_data = dict(client.compute(sentiment_tasks))
        
        return sentiment_data
    
    finally:
        # Don't close the client here to allow reuse
        pass

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
    
    # Get Dask client for parallel processing
    client = get_dask_client()
    
    try:
        # Define a function to process each cluster
        @delayed
        def process_cluster_keywords(cluster_id, data):
            """Process keyword data for a single cluster."""
            return cluster_id, {
                'top_keywords': data['top_keywords'],
                'size': data['size']
            }
        
        # Schedule parallel processing of cluster keywords
        keyword_tasks = []
        for cluster_id, data in clustering_results['cluster_analysis'].items():
            keyword_tasks.append(process_cluster_keywords(cluster_id, data))
        
        # Compute all keyword data in parallel
        keyword_data = dict(client.compute(keyword_tasks))
        
        return keyword_data
    
    finally:
        # Don't close the client here to allow reuse
        pass

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
    
    # Get Dask client for parallel processing
    client = get_dask_client()
    
    try:
        # Define a function to process each cluster
        @delayed
        def process_cluster_hashtags(cluster_id, data):
            """Process hashtag data for a single cluster."""
            return cluster_id, {
                'top_hashtags': data['top_hashtags'],
                'size': data['size']
            }
        
        # Schedule parallel processing of cluster hashtags
        hashtag_tasks = []
        for cluster_id, data in clustering_results['cluster_analysis'].items():
            hashtag_tasks.append(process_cluster_hashtags(cluster_id, data))
        
        # Compute all hashtag data in parallel
        hashtag_data = dict(client.compute(hashtag_tasks))
        
        return hashtag_data
    
    finally:
        # Don't close the client here to allow reuse
        pass

def get_all_cluster_analyses():
    """
    Get a list of all available cluster analyses in parallel.
    
    Returns:
        List of dictionaries with analysis metadata
    """
    analyses_dir = 'app/data/analysis/'
    
    # Create directory if it doesn't exist
    if not os.path.exists(analyses_dir):
        os.makedirs(analyses_dir)
    
    # Get Dask client for parallel processing
    client = get_dask_client()
    
    try:
        # Get all JSON files in the directory
        analysis_files = [f for f in os.listdir(analyses_dir) 
                         if f.startswith('user_clustering_') and f.endswith('.json')]
        
        # Define a function to process each analysis file in parallel
        @delayed
        def process_analysis_file(filename):
            """Process a single analysis file."""
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
            
            return {
                'filepath': filepath,
                'method': method,
                'timestamp': formatted_timestamp,
                'n_clusters': analysis_data['n_clusters'],
                'user_count': analysis_data['user_count']
            }
        
        # Schedule parallel processing of analysis files
        analysis_tasks = []
        for filename in analysis_files:
            analysis_tasks.append(process_analysis_file(filename))
        
        # Compute all analysis metadata in parallel
        analyses = client.compute(analysis_tasks)
        
        # Sort by timestamp, most recent first
        analyses = sorted(analyses, key=lambda x: x['timestamp'], reverse=True)
        
        return analyses
    
    finally:
        # Don't close the client here to allow reuse
        pass

def shutdown_parallel_processing():
    """Shut down the Dask client when done with all processing."""
    close_dask_client()