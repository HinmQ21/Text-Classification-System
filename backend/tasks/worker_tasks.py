import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from rq import get_current_job

from services.text_classifier import TextClassifierService
from services.language_detector import LanguageDetectorService
from services.csv_processor import CSVProcessorService
from models.database import get_db, save_classification_result, User
from models.schemas import TextClassificationRequest, CSVUploadRequest

logger = logging.getLogger(__name__)

# Initialize services (these will be initialized by workers)
text_classifier = None
language_detector = None
csv_processor = None

async def init_services():
    """Initialize services for worker"""
    global text_classifier, language_detector, csv_processor
    
    if text_classifier is None:
        text_classifier = TextClassifierService()
        await text_classifier.initialize()
        
    if language_detector is None:
        language_detector = LanguageDetectorService()
        
    if csv_processor is None:
        csv_processor = CSVProcessorService()
        csv_processor.initialize(text_classifier, language_detector)

def classify_text_task(
    text: str,
    model_type: str,
    temperature: float = 0.7,
    model_selection: str = "all",
    user_id: Optional[int] = None,
    enable_translation: bool = True
) -> Dict[str, Any]:
    """Task function for single text classification"""
    try:
        # Initialize services
        asyncio.get_event_loop().run_until_complete(init_services())
        
        # Get current job for progress tracking
        job = get_current_job()
        if job:
            job.meta['progress'] = {'status': 'detecting_language', 'step': 1, 'total_steps': 3}
            job.save_meta()
        
        # Detect language
        detected_language = language_detector.detect(text)
        
        if job:
            job.meta['progress'] = {'status': 'classifying', 'step': 2, 'total_steps': 3}
            job.save_meta()
        
        # Classify text
        result = asyncio.get_event_loop().run_until_complete(
            text_classifier.classify(
                text=text,
                model_type=model_type,
                language=detected_language,
                temperature=temperature,
                model_selection=model_selection,
                enable_translation=enable_translation
            )
        )
        
        if job:
            job.meta['progress'] = {'status': 'saving_result', 'step': 3, 'total_steps': 3}
            job.save_meta()
        
        # Save to database if user is provided
        if user_id:
            try:
                db = next(get_db())
                save_classification_result(
                    db=db,
                    text=text,
                    model_type=model_type,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    language=detected_language,
                    processing_time=result.get("processing_time", 0),
                    user_id=user_id
                )
                db.close()
            except Exception as e:
                logger.error(f"Failed to save classification result: {str(e)}")
        
        # Prepare response
        task_result = {
            "text": text,
            "model_type": model_type,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "temperature": result["temperature"],
            "language": detected_language,
            "processing_time": result.get("processing_time", 0),
            "timestamp": datetime.now().isoformat(),
            "is_ensemble": result.get("is_ensemble", False),
            "models_used": result.get("models_used", []),
            "individual_results": result.get("individual_results"),
            "status": "completed"
        }
        
        if job:
            job.meta['progress'] = {'status': 'completed', 'step': 3, 'total_steps': 3}
            job.save_meta()
        
        logger.info(f"Classification task completed for job {job.id if job else 'unknown'}")
        return task_result
        
    except Exception as e:
        logger.error(f"Classification task failed: {str(e)}")
        if job:
            job.meta['progress'] = {'status': 'failed', 'error': str(e)}
            job.save_meta()
        raise

def batch_classify_task(
    texts: List[str],
    model_type: str,
    temperature: float = 0.7,
    model_selection: str = "all",
    enable_translation: bool = True
) -> Dict[str, Any]:
    """Task function for batch text classification"""
    try:
        # Initialize services
        asyncio.get_event_loop().run_until_complete(init_services())
        
        job = get_current_job()
        total_texts = len(texts)
        results = []
        
        if job:
            job.meta['progress'] = {
                'status': 'processing', 
                'processed': 0, 
                'total': total_texts,
                'percentage': 0
            }
            job.save_meta()
        
        for i, text in enumerate(texts):
            try:
                # Detect language
                detected_language = language_detector.detect(text)
                
                # Classify text
                result = asyncio.get_event_loop().run_until_complete(
                    text_classifier.classify(
                        text=text,
                        model_type=model_type,
                        language=detected_language,
                        temperature=temperature,
                        model_selection=model_selection,
                        enable_translation=enable_translation
                    )
                )
                
                results.append({
                    "text": text,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "language": detected_language,
                    "is_ensemble": result.get("is_ensemble", False),
                    "models_used": result.get("models_used", [])
                })
                
                # Update progress
                if job:
                    processed = i + 1
                    percentage = int((processed / total_texts) * 100)
                    job.meta['progress'] = {
                        'status': 'processing',
                        'processed': processed,
                        'total': total_texts,
                        'percentage': percentage
                    }
                    job.save_meta()
                
            except Exception as e:
                logger.error(f"Failed to classify text {i}: {str(e)}")
                results.append({
                    "text": text,
                    "error": str(e),
                    "status": "failed"
                })
        
        task_result = {
            "model_type": model_type,
            "total_processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        if job:
            job.meta['progress'] = {
                'status': 'completed',
                'processed': total_texts,
                'total': total_texts,
                'percentage': 100
            }
            job.save_meta()
        
        logger.info(f"Batch classification task completed for {total_texts} texts")
        return task_result
        
    except Exception as e:
        logger.error(f"Batch classification task failed: {str(e)}")
        if job:
            job.meta['progress'] = {'status': 'failed', 'error': str(e)}
            job.save_meta()
        raise

def csv_processing_task(
    file_content: str,
    csv_request_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Task function for CSV processing"""
    try:
        # Initialize services
        asyncio.get_event_loop().run_until_complete(init_services())
        
        job = get_current_job()
        if job:
            job.meta['progress'] = {'status': 'parsing_csv', 'step': 1, 'total_steps': 3}
            job.save_meta()
        
        # Recreate CSVUploadRequest from dict
        csv_request = CSVUploadRequest(**csv_request_dict)
        
        # Process CSV using existing csv_processor logic
        # We'll need to adapt this to work synchronously
        db = next(get_db())
        
        if job:
            job.meta['progress'] = {'status': 'processing_texts', 'step': 2, 'total_steps': 3}
            job.save_meta()
        
        # Start processing (this will need to be adapted from async to sync)
        job_id = asyncio.get_event_loop().run_until_complete(
            csv_processor.start_csv_processing(file_content, csv_request, db)
        )
        
        if job:
            job.meta['progress'] = {'status': 'getting_results', 'step': 3, 'total_steps': 3}
            job.save_meta()
        
        # Wait for processing to complete and get results
        # This is a simplified version - in practice you might want to poll the status
        results = csv_processor.get_job_results(job_id, db)
        
        db.close()
        
        if job:
            job.meta['progress'] = {'status': 'completed', 'step': 3, 'total_steps': 3}
            job.save_meta()
        
        logger.info(f"CSV processing task completed for job {job.id if job else 'unknown'}")
        return results.__dict__ if results else {"status": "failed", "error": "No results"}
        
    except Exception as e:
        logger.error(f"CSV processing task failed: {str(e)}")
        if job:
            job.meta['progress'] = {'status': 'failed', 'error': str(e)}
            job.save_meta()
        raise

# Health check task for workers
def health_check_task() -> Dict[str, Any]:
    """Simple health check task to verify worker is functioning"""
    try:
        job = get_current_job()
        return {
            "status": "healthy",
            "worker_id": job.id if job else "unknown",
            "timestamp": datetime.now().isoformat(),
            "message": "Worker is functioning correctly"
        }
    except Exception as e:
        logger.error(f"Health check task failed: {str(e)}")
        raise 