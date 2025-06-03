#!/usr/bin/env python3
"""
Script to start multiple RQ workers for different queues
"""

import subprocess
import sys
import time
import signal
import os
from multiprocessing import Process
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkerManager:
    def __init__(self):
        self.workers = []
        self.running = False

    def start_worker_process(self, queue_name, worker_name):
        """Start a worker process for a specific queue"""
        try:
            cmd = [
                sys.executable, 
                "worker.py", 
                "--queue", queue_name,
                "--verbose"
            ]
            
            logger.info(f"Starting {worker_name} worker for queue: {queue_name}")
            
            # Create log file for this worker
            log_file = f"worker_{queue_name}.log"
            
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            
            # Give the process a moment to start and check if it's still running
            time.sleep(2)
            
            if process.poll() is None:
                logger.info(f"Successfully started {worker_name} worker (PID: {process.pid})")
                return process
            else:
                # Process has already exited, check logs
                with open(log_file, "r") as f:
                    error_output = f.read()
                logger.error(f"Failed to start {worker_name} worker. Error output:")
                logger.error(error_output)
                return None
            
        except Exception as e:
            logger.error(f"Failed to start {worker_name} worker: {str(e)}")
            return None

    def start_all_workers(self):
        """Start all workers for different queues"""
        logger.info("Starting all RQ workers...")
        
        # Worker configurations
        worker_configs = [
            ("classification", "Classification"),
            ("batch_processing", "Batch Processing"),
            ("csv_processing", "CSV Processing")
        ]
        
        for queue_name, worker_name in worker_configs:
            process = self.start_worker_process(queue_name, worker_name)
            if process:
                self.workers.append({
                    'process': process,
                    'name': worker_name,
                    'queue': queue_name
                })
                time.sleep(1)  # Small delay between starting workers
        
        if len(self.workers) == 0:
            logger.error("Failed to start any workers!")
            return False
            
        logger.info(f"Started {len(self.workers)} workers")
        self.running = True
        return True

    def monitor_workers(self):
        """Monitor worker processes and restart if they crash"""
        logger.info("Monitoring workers...")
        
        try:
            while self.running:
                time.sleep(10)  # Check every 10 seconds (increased from 5)
                
                for i, worker in enumerate(self.workers):
                    process = worker['process']
                    
                    # Check if process is still running
                    if process.poll() is not None:
                        logger.warning(f"Worker {worker['name']} (PID: {process.pid}) has stopped")
                        
                        # Check worker log for errors
                        log_file = f"worker_{worker['queue']}.log"
                        try:
                            with open(log_file, "r") as f:
                                last_lines = f.read().split('\n')[-10:]  # Get last 10 lines
                                logger.error(f"Last log entries for {worker['name']}:")
                                for line in last_lines:
                                    if line.strip():
                                        logger.error(f"  {line}")
                        except Exception:
                            pass
                        
                        # Restart the worker
                        logger.info(f"Restarting {worker['name']} worker...")
                        new_process = self.start_worker_process(worker['queue'], worker['name'])
                        
                        if new_process:
                            self.workers[i]['process'] = new_process
                            logger.info(f"Successfully restarted {worker['name']} worker")
                        else:
                            logger.error(f"Failed to restart {worker['name']} worker")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping workers...")
            self.stop_all_workers()

    def stop_all_workers(self):
        """Stop all worker processes"""
        logger.info("Stopping all workers...")
        self.running = False
        
        for worker in self.workers:
            process = worker['process']
            try:
                logger.info(f"Stopping {worker['name']} worker (PID: {process.pid})")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"Successfully stopped {worker['name']} worker")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {worker['name']} worker")
                    process.kill()
                    
            except Exception as e:
                logger.error(f"Error stopping {worker['name']} worker: {str(e)}")
        
        logger.info("All workers stopped")

    def get_worker_status(self):
        """Get status of all workers"""
        status = []
        for worker in self.workers:
            process = worker['process']
            is_running = process.poll() is None
            status.append({
                'name': worker['name'],
                'queue': worker['queue'],
                'pid': process.pid,
                'running': is_running
            })
        return status

def test_single_worker():
    """Test starting a single worker to verify everything works"""
    logger.info("Testing single worker startup...")
    
    try:
        cmd = [sys.executable, "worker.py", "--queue", "classification", "--verbose", "--burst"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("Single worker test passed")
            return True
        else:
            logger.error("Single worker test failed:")
            logger.error(f"Exit code: {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.info("Single worker test timeout (this is normal for non-burst mode)")
        return True
    except Exception as e:
        logger.error(f"Single worker test error: {str(e)}")
        return False

def main():
    """Main function to start and manage workers"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ Worker Manager')
    parser.add_argument(
        '--mode',
        choices=['start', 'monitor', 'test'],
        default='monitor',
        help='Mode to run: start (start once), monitor (start and monitor), or test (test single worker)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers per queue (default: 1)'
    )
    
    args = parser.parse_args()
    
    if args.verbose if hasattr(args, 'verbose') else False:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test mode
    if args.mode == 'test':
        success = test_single_worker()
        sys.exit(0 if success else 1)
    
    manager = WorkerManager()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        manager.stop_all_workers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start workers
        success = manager.start_all_workers()
        if not success:
            logger.error("Failed to start workers. Exiting.")
            sys.exit(1)
        
        if args.mode == 'monitor':
            # Monitor and restart workers if needed
            manager.monitor_workers()
        else:
            # Just start and exit
            logger.info("Workers started. Use 'python worker.py' to start individual workers.")
            
    except Exception as e:
        logger.error(f"Error in worker manager: {str(e)}")
        manager.stop_all_workers()
        sys.exit(1)

if __name__ == '__main__':
    main()