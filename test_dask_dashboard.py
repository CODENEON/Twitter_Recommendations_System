#!/usr/bin/env python3
"""
Twitter Recommendation System - Dask Dashboard Test

This script creates long-running visible tasks in the Dask dashboard
to demonstrate the parallel processing system.
"""

import time
import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_twitter_recommendation_dask():
    """Test Dask dashboard with visible long-running tasks."""
    logger.info("=" * 80)
    logger.info("TWITTER RECOMMENDATION SYSTEM - DASK DASHBOARD VISIBILITY TEST")
    logger.info("=" * 80)
    
    try:
        # Import Dask with verbose error reporting if it fails
        try:
            import dask
            from dask.distributed import Client, LocalCluster, wait
            from dask import delayed
            import dask.array as da
            
            logger.info(f"Successfully imported Dask version: {dask.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import Dask: {e}")
            logger.error("Please install required packages with: pip install 'dask[complete]' distributed")
            return False
            
        # Create a local cluster with explicit settings
        logger.info("Starting local Dask cluster...")
        cluster = LocalCluster(
            n_workers=5,                # Use 5 workers
            threads_per_worker=4,       # 4 threads per worker
            memory_limit='2GB',         # 2GB memory limit per worker
            dashboard_address=':8787',  # Explicit dashboard port
            silence_logs=logging.WARNING  # Reduce Dask's internal logging
        )
        logger.info(f"Local cluster started with {cluster.scheduler.identity()['workers']} workers")
        
        # Create a client
        logger.info("Creating Dask client...")
        client = Client(cluster)
        dashboard_url = client.dashboard_link
        
        # Print dashboard URL very clearly
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"DASK DASHBOARD URL: {dashboard_url}")
        logger.info("Open this URL in your web browser to monitor task execution")
        logger.info("=" * 80)
        logger.info("")
        
        # Also print to stdout directly (in case logging is redirected)
        print("\n" + "=" * 80)
        print(f"DASK DASHBOARD URL: {dashboard_url}")
        print("Open this URL in your web browser to monitor task execution")
        print("=" * 80 + "\n")
        
        # Wait a moment to ensure client is connected
        time.sleep(2)
        
        # ----------------------------------------
        # Test 1: Visible Long-Running Tasks
        # ----------------------------------------
        logger.info("Creating long-running tasks to visualize in dashboard...")
        
        @delayed
        def long_running_task(task_id, delay):
            """A deliberately slow function that will be visible in the dashboard."""
            logger.info(f"Starting task {task_id} with delay {delay}s")
            time.sleep(delay)  # Sleep to make it very visible
            
            # Do some actual computation to show CPU activity
            result = 0
            for i in range(1000000):
                result += i * np.sin(i * 0.01)
                
            logger.info(f"Completed task {task_id}")
            return result
        
        # Create a complex task graph with dependencies
        logger.info("Building task graph with dependencies...")
        
        # Create first layer of tasks
        layer1_tasks = []
        for i in range(20):
            # Tasks with varying durations (5-15 seconds)
            layer1_tasks.append(long_running_task(f"layer1-{i}", 5 + (i % 11)))
        
        # Create second layer of tasks that depend on first layer
        layer2_tasks = []
        for i in range(10):
            # Each layer2 task depends on two layer1 tasks
            input1 = layer1_tasks[i]
            input2 = layer1_tasks[i+10]
            
            # Create a delayed task that uses results from previous tasks
            @delayed
            def combine_results(x, y, task_id):
                logger.info(f"Starting combined task {task_id}")
                time.sleep(8)  # Make it visible
                result = x + y + np.random.random()
                logger.info(f"Completed combined task {task_id}")
                return result
                
            layer2_tasks.append(combine_results(input1, input2, f"layer2-{i}"))
        
        # Create a final aggregation task
        @delayed
        def final_aggregation(inputs):
            logger.info("Starting final aggregation")
            time.sleep(10)  # Make it visible
            result = sum(inputs)
            logger.info(f"Final result: {result}")
            return result
            
        final_task = final_aggregation(layer2_tasks)
        
        # ----------------------------------------
        # Test 2: Array Computation That Should Be Visible
        # ----------------------------------------
        logger.info("Creating large array computations...")
        
        # Create a large random array (will show memory usage)
        x = da.random.random((8000, 8000), chunks=(1000, 1000))
        logger.info("Created large random dask array")
        
        # Perform calculations designed to be visible
        y = x + x.T
        z = da.sin(y) + da.cos(y)
        mean_value = z.mean()
        
        # Submit all tasks to the scheduler asynchronously
        logger.info("Submitting all tasks to the scheduler...")
        future = client.compute(final_task)
        array_future = client.compute(mean_value)
        
        # Tell user to look at the dashboard now
        logger.info("")
        logger.info("=" * 80)
        logger.info("TASKS ARE NOW RUNNING - CHECK THE DASHBOARD AT:")
        logger.info(f"{dashboard_url}")
        logger.info("You should see task activity in the dashboard for several minutes")
        logger.info("=" * 80)
        logger.info("")
        
        # Wait for tasks to complete while giving status updates
        logger.info("Waiting for tasks to complete (this will take several minutes)...")
        task_complete = False
        array_complete = False
        
        start_time = time.time()
        while not (task_complete and array_complete):
            current_time = time.time() - start_time
            sys.stdout.write(f"\rExecution time: {current_time:.1f}s - Check dashboard for visualization")
            sys.stdout.flush()
            
            if not task_complete and future.done():
                task_complete = True
                logger.info("\nTask graph computation complete!")
                
            if not array_complete and array_future.done():
                array_complete = True
                logger.info("\nArray computation complete!")
                
            # Brief pause to allow cancellation
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user - will attempt graceful shutdown")
                break
        
        # Get and display results
        if task_complete:
            try:
                result = future.result()
                logger.info(f"Final computation result: {result}")
            except Exception as e:
                logger.error(f"Task computation failed: {e}")
                
        if array_complete:
            try:
                array_result = array_future.result()
                logger.info(f"Array computation result: {array_result}")
            except Exception as e:
                logger.error(f"Array computation failed: {e}")
        
        # Simulate the Twitter recommendation workload in continuous mode for visualization
        logger.info("\nStarting continuous Twitter recommendation simulation for dashboard visualization...")
        
        # Create a simulation of the recommendation system's workload
        @delayed
        def simulate_sentiment_analysis(batch_id):
            logger.info(f"Sentiment analysis batch {batch_id}")
            time.sleep(3)  # Simulate processing time
            return f"sentiment_batch_{batch_id}"
            
        @delayed
        def simulate_user_recommendation(user_id):
            logger.info(f"Recommendation for user {user_id}")
            time.sleep(5)  # Simulate processing time
            return f"recommendations_for_user_{user_id}"
            
        @delayed
        def simulate_trending_topics():
            logger.info("Updating trending topics")
            time.sleep(7)  # Simulate processing time
            return "trending_topics_updated"
        
        # Keep the dashboard active by periodically submitting new tasks
        end_time = time.time() + 300  # Run for 5 minutes
        
        while time.time() < end_time:
            batch_tasks = [simulate_sentiment_analysis(i) for i in range(5)]
            user_tasks = [simulate_user_recommendation(i) for i in range(3)]
            trending_task = simulate_trending_topics()
            
            # Submit tasks
            batch_futures = client.compute(batch_tasks)
            user_futures = client.compute(user_tasks)
            trending_future = client.compute(trending_task)
            
            # Display status
            remaining = int(end_time - time.time())
            sys.stdout.write(f"\rSimulation running - {remaining}s remaining. Check dashboard for activity.")
            sys.stdout.flush()
            
            # Wait before submitting next batch
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user - will attempt graceful shutdown")
                break
        
        # Test successful
        logger.info("\nTwitter recommendation system Dask dashboard test complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return False
    
    finally:
        # Clean shutdown
        try:
            if 'client' in locals():
                logger.info("Closing Dask client...")
                client.close()
            if 'cluster' in locals():
                logger.info("Shutting down Dask cluster...")
                cluster.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    test_twitter_recommendation_dask()