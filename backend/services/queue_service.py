import json
import logging
from typing import Dict, Any, Optional, List
from rq.job import Job
from rq.exceptions import NoSuchJobError
from datetime import datetime, timedelta

from config.redis_config import (
    redis_conn, 
    classification_queue, 
    csv_processing_queue, 
    batch_processing_queue,
    QUEUE_NAMES
)

logger = logging.getLogger(__name__)

class QueueService:
    """Service for managing RQ job queues"""
    
    def __init__(self):
        self.queues = {
            QUEUE_NAMES['CLASSIFICATION']: classification_queue,
            QUEUE_NAMES['CSV_PROCESSING']: csv_processing_queue,
            QUEUE_NAMES['BATCH_PROCESSING']: batch_processing_queue
        }
    
    def enqueue_classification(self, task_func, *args, **kwargs) -> str:
        """Enqueue a text classification task"""
        try:
            job = classification_queue.enqueue(task_func, *args, **kwargs)
            logger.info(f"Enqueued classification job: {job.id}")
            return job.id
        except Exception as e:
            logger.error(f"Failed to enqueue classification job: {str(e)}")
            raise
    
    def enqueue_batch_processing(self, task_func, *args, **kwargs) -> str:
        """Enqueue a batch processing task"""
        try:
            job = batch_processing_queue.enqueue(task_func, *args, **kwargs)
            logger.info(f"Enqueued batch processing job: {job.id}")
            return job.id
        except Exception as e:
            logger.error(f"Failed to enqueue batch processing job: {str(e)}")
            raise
    
    def enqueue_csv_processing(self, task_func, *args, **kwargs) -> str:
        """Enqueue a CSV processing task"""
        try:
            job = csv_processing_queue.enqueue(task_func, *args, **kwargs)
            logger.info(f"Enqueued CSV processing job: {job.id}")
            return job.id
        except Exception as e:
            logger.error(f"Failed to enqueue CSV processing job: {str(e)}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job by ID"""
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            
            status_info = {
                "job_id": job_id,
                "status": job.get_status(),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "result": job.result if job.is_finished else None,
                "failure_info": job.exc_info if job.is_failed else None,
                "progress": self._get_job_progress(job),
                "queue_name": job.origin
            }
            
            if job.is_failed:
                status_info["error"] = str(job.exc_info) if job.exc_info else "Unknown error"
            
            return status_info
            
        except NoSuchJobError:
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": "Job not found"
            }
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get result of a completed job"""
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_finished:
                return job.result
            return None
        except NoSuchJobError:
            return None
        except Exception as e:
            logger.error(f"Error getting job result: {str(e)}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            job.cancel()
            logger.info(f"Cancelled job: {job_id}")
            return True
        except NoSuchJobError:
            logger.warning(f"Job not found for cancellation: {job_id}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return False
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about all queues"""
        queue_info = {}
        
        for name, queue in self.queues.items():
            try:
                queue_info[name] = {
                    "name": name,
                    "length": len(queue),
                    "failed_jobs": queue.failed_job_registry.count,
                    "scheduled_jobs": queue.scheduled_job_registry.count,
                    "started_jobs": queue.started_job_registry.count,
                    "finished_jobs": queue.finished_job_registry.count
                }
            except Exception as e:
                logger.error(f"Error getting queue info for {name}: {str(e)}")
                queue_info[name] = {"error": str(e)}
        
        return queue_info
    
    def clean_old_jobs(self, days: int = 7) -> Dict[str, int]:
        """Clean up old completed jobs"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_counts = {}
        
        for name, queue in self.queues.items():
            try:
                # Clean finished jobs older than cutoff_date
                finished_jobs = queue.finished_job_registry.get_job_ids()
                cleaned_count = 0
                
                for job_id in finished_jobs:
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.ended_at and job.ended_at < cutoff_date:
                            job.delete()
                            cleaned_count += 1
                    except NoSuchJobError:
                        continue
                
                cleaned_counts[name] = cleaned_count
                logger.info(f"Cleaned {cleaned_count} old jobs from queue {name}")
                
            except Exception as e:
                logger.error(f"Error cleaning queue {name}: {str(e)}")
                cleaned_counts[name] = 0
        
        return cleaned_counts
    
    def _get_job_progress(self, job) -> Optional[Dict[str, Any]]:
        """Get job progress information if available"""
        try:
            # Try to get progress from job meta
            if hasattr(job, 'meta') and job.meta:
                return job.meta.get('progress')
            return None
        except Exception:
            return None

# Global queue service instance
queue_service = QueueService() 