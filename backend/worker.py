#!/usr/bin/env python3
"""
RQ Worker for text classification system
Run this script to start processing jobs from the Redis queues
"""

import sys
import os
import logging
from rq import Worker, Connection

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('worker.log')
    ]
)
logger = logging.getLogger(__name__)

def run_worker(queue_names=None, burst=False):
    """
    Run an RQ worker
    
    Args:
        queue_names: List of queue names to listen to. If None, listens to all queues
        burst: If True, worker will exit when all jobs are processed
    """
    try:
        from config.redis_config import redis_conn, QUEUE_NAMES
        from services.queue_service import queue_service
        
        if queue_names is None:
            queue_names = list(QUEUE_NAMES.values())
        
        logger.info(f"Starting worker for queues: {queue_names}")
        
        with Connection(redis_conn):
            worker = Worker(
                queue_names, 
                connection=redis_conn,
                name=f"worker-{os.getpid()}"
            )
            
            logger.info(f"Worker {worker.name} started successfully")
            logger.info(f"Worker PID: {os.getpid()}")
            logger.info(f"Listening on queues: {queue_names}")
            
            # Start the worker
            worker.work(burst=burst, with_scheduler=True)
            
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    logger.info("Worker stopped")

def run_classification_worker():
    """Run worker for classification queue only"""
    from config.redis_config import QUEUE_NAMES
    run_worker([QUEUE_NAMES['CLASSIFICATION']])

def run_batch_worker():
    """Run worker for batch processing queue only"""
    from config.redis_config import QUEUE_NAMES
    run_worker([QUEUE_NAMES['BATCH_PROCESSING']])

def run_csv_worker():
    """Run worker for CSV processing queue only"""
    from config.redis_config import QUEUE_NAMES
    run_worker([QUEUE_NAMES['CSV_PROCESSING']])

def run_all_queues_worker():
    """Run worker for all queues"""
    run_worker()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ Worker for Text Classification System')
    parser.add_argument(
        '--queue', 
        choices=['classification', 'batch_processing', 'csv_processing', 'all'],
        default='all',
        help='Queue to process (default: all)'
    )
    parser.add_argument(
        '--burst',
        action='store_true',
        help='Run in burst mode (exit when queue is empty)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test Redis connection
    try:
        from config.redis_config import redis_conn, QUEUE_NAMES
        redis_conn.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        sys.exit(1)
    
    # Run the appropriate worker
    if args.queue == 'classification':
        run_worker([QUEUE_NAMES['CLASSIFICATION']], burst=args.burst)
    elif args.queue == 'batch_processing':
        run_worker([QUEUE_NAMES['BATCH_PROCESSING']], burst=args.burst)
    elif args.queue == 'csv_processing':
        run_worker([QUEUE_NAMES['CSV_PROCESSING']], burst=args.burst)
    else:
        run_worker(burst=args.burst) 