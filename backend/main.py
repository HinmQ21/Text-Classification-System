from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import desc
import math

# Load environment variables
load_dotenv()

# Import our modules
from models.database import init_db, get_db, User, ClassificationResult
from models.schemas import (
    TextClassificationRequest, TextClassificationResponse, BatchRequest,
    CSVUploadRequest, CSVBatchResponse, BatchProcessingStatus,
    UserCreate, UserLogin, Token, UserResponse, QueryHistoryResponse, QueryHistoryItem
)
from services.text_classifier import TextClassifierService
from services.language_detector import LanguageDetectorService
from services.csv_processor import CSVProcessorService
from services.auth_service import auth_service
from services.auth_dependencies import get_current_user_optional, get_current_user_required
from services.queue_service import queue_service
from tasks.worker_tasks import classify_text_task, batch_classify_task, csv_processing_task, health_check_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="A demo API for text classification with multi-language support and Redis queue processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
text_classifier = TextClassifierService()
language_detector = LanguageDetectorService()
csv_processor = CSVProcessorService()

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    init_db()
    await text_classifier.initialize()
    csv_processor.initialize(text_classifier, language_detector)
    logger.info("Application started successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Text Classification API is running",
        "timestamp": datetime.now().isoformat(),
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check Redis connection
        redis_healthy = True
        try:
            from config.redis_config import redis_conn
            redis_conn.ping()
        except Exception:
            redis_healthy = False
        
        return {
            "status": "healthy",
            "services": {
                "text_classifier": text_classifier.is_ready(),
                "language_detector": True,
                "database": True,
                "redis": redis_healthy,
                "queue_service": True
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Original synchronous endpoint (kept for backward compatibility)
@app.post("/classify", response_model=TextClassificationResponse)
async def classify_text(
    request: TextClassificationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Classify a single text input (synchronous)
    Supports: sentiment, spam, topic classification
    Works for both authenticated and anonymous users
    """
    try:
        # Detect language
        detected_language = language_detector.detect(request.text)

        # Classify text with temperature parameter and model selection
        result = await text_classifier.classify(
            text=request.text,
            model_type=request.model_type,
            language=detected_language,
            temperature=request.temperature,
            model_selection=request.model_selection,
            enable_translation=request.enable_translation
        )

        # Save result to database if user is authenticated
        if current_user:
            from models.database import save_classification_result
            save_classification_result(
                db=db,
                text=request.text,
                model_type=request.model_type,
                prediction=result["prediction"],
                confidence=result["confidence"],
                language=detected_language,
                processing_time=result.get("processing_time", 0),
                user_id=current_user.id
            )

        return TextClassificationResponse(
            text=request.text,
            model_type=request.model_type,
            prediction=result["prediction"],
            confidence=result["confidence"],
            all_scores=result["all_scores"],
            temperature=result["temperature"],
            language=detected_language,
            processing_time=result.get("processing_time", 0),
            timestamp=datetime.now(),
            is_ensemble=result.get("is_ensemble", False),
            models_used=result.get("models_used", []),
            individual_results=result.get("individual_results")
        )

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# New asynchronous endpoint using queue
@app.post("/classify/async")
async def classify_text_async(
    request: TextClassificationRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Classify a single text input asynchronously using Redis queue
    Returns job_id to track processing status
    """
    try:
        # Enqueue the classification task
        job_id = queue_service.enqueue_classification(
            classify_text_task,
            text=request.text,
            model_type=request.model_type,
            temperature=request.temperature,
            model_selection=request.model_selection,
            user_id=current_user.id if current_user else None,
            enable_translation=request.enable_translation
        )
        
        return {
            "job_id": job_id,
            "message": "Classification task queued successfully",
            "status": "queued",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to queue classification task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue classification: {str(e)}")

@app.post("/classify/batch")
async def classify_batch(
    request: BatchRequest,
    db = Depends(get_db)
):
    """
    Classify multiple texts in batch (synchronous - kept for backward compatibility)
    """
    try:
        results = []
        
        for text in request.texts:
            # Detect language
            detected_language = language_detector.detect(text)
            
            # Classify text with model selection
            result = await text_classifier.classify(
                text=text,
                model_type=request.model_type,
                language=detected_language,
                temperature=request.temperature,
                model_selection=request.model_selection,
                enable_translation=request.enable_translation
            )
            
            results.append({
                "text": text,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "language": detected_language,
                "is_ensemble": result.get("is_ensemble", False),
                "models_used": result.get("models_used", [])
            })
        
        return {
            "model_type": request.model_type,
            "total_processed": len(results),
            "results": results,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

# New asynchronous batch endpoint
@app.post("/classify/batch/async")
async def classify_batch_async(request: BatchRequest):
    """
    Classify multiple texts in batch asynchronously using Redis queue
    Returns job_id to track processing status
    """
    try:
        # Enqueue the batch classification task
        job_id = queue_service.enqueue_batch_processing(
            batch_classify_task,
            texts=request.texts,
            model_type=request.model_type,
            temperature=request.temperature,
            model_selection=request.model_selection,
            enable_translation=request.enable_translation
        )
        
        return {
            "job_id": job_id,
            "message": "Batch classification task queued successfully",
            "status": "queued",
            "total_texts": len(request.texts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to queue batch classification task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue batch classification: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of available classification models"""
    try:
        available_models = text_classifier.get_available_models()
        
        model_list = []
        for model_type, models in available_models.items():
            if model_type == "sentiment":
                description = "Sentiment Analysis (Positive/Negative/Neutral)"
                languages = ["en", "multilingual", "auto"]
            elif model_type == "spam":
                description = "Spam Detection (Spam/Not Spam)"
                languages = ["en", "auto"]
            elif model_type == "topic":
                description = "Topic Classification"
                languages = ["en", "auto"]
            else:
                description = f"{model_type.title()} Classification"
                languages = ["en", "auto"]
            
            model_list.append({
                "name": model_type,
                "description": description,
                "languages": languages,
                "available_models": models
            })
        
        return {
            "available_models": model_list
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@app.post("/classify/csv")
async def upload_csv_for_classification(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    batch_size: int = Form(default=10),
    text_column: str = Form(default="text"),
    model_selection: str = Form(default="all"),
    enable_translation: bool = Form(default=True),
    db = Depends(get_db)
):
    """
    Upload CSV file for batch text classification (synchronous - kept for backward compatibility)
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Read file content
        file_content = await file.read()
        file_content_str = file_content.decode('utf-8')
        
        # Create request object
        # Parse model_selection (can be comma-separated string)
        parsed_model_selection = model_selection
        if model_selection != "all" and "," in model_selection:
            parsed_model_selection = model_selection.split(",")
        
        csv_request = CSVUploadRequest(
            model_type=model_type,
            batch_size=batch_size,
            text_column=text_column,
            model_selection=parsed_model_selection,
            enable_translation=enable_translation
        )

        # Start processing
        job_id = await csv_processor.start_csv_processing(file_content_str, csv_request, db)

        return {
            "job_id": job_id,
            "message": "CSV processing started",
            "status": "processing"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CSV upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

# New asynchronous CSV endpoint
@app.post("/classify/csv/async")
async def upload_csv_for_classification_async(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    batch_size: int = Form(default=10),
    text_column: str = Form(default="text"),
    model_selection: str = Form(default="all"),
    enable_translation: bool = Form(default=True)
):
    """
    Upload CSV file for batch text classification asynchronously using Redis queue
    Returns job_id to track processing status
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Read file content
        file_content = await file.read()
        file_content_str = file_content.decode('utf-8')
        
        # Parse model_selection (can be comma-separated string)
        parsed_model_selection = model_selection
        if model_selection != "all" and "," in model_selection:
            parsed_model_selection = model_selection.split(",")
        
        # Create request object as dict for serialization
        csv_request_dict = {
            "model_type": model_type,
            "batch_size": batch_size,
            "text_column": text_column,
            "model_selection": parsed_model_selection,
            "enable_translation": enable_translation
        }

        # Enqueue the CSV processing task
        job_id = queue_service.enqueue_csv_processing(
            csv_processing_task,
            file_content=file_content_str,
            csv_request_dict=csv_request_dict
        )

        return {
            "job_id": job_id,
            "message": "CSV processing task queued successfully",
            "status": "queued",
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to queue CSV processing task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue CSV processing: {str(e)}")

@app.get("/classify/csv/status/{job_id}", response_model=BatchProcessingStatus)
async def get_csv_processing_status(
    job_id: str,
    db = Depends(get_db)
):
    """
    Get the current status of a CSV processing job
    """
    try:
        status = csv_processor.get_job_status(job_id, db)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        return status

    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/classify/csv/results/{job_id}", response_model=CSVBatchResponse)
async def get_csv_processing_results(
    job_id: str,
    db = Depends(get_db)
):
    """
    Get the complete results of a CSV processing job
    """
    try:
        results = csv_processor.get_job_results(job_id, db)
        if not results:
            raise HTTPException(status_code=404, detail="Job not found")

        return results

    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job results: {str(e)}")

# New queue management endpoints
@app.get("/queue/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of any job by ID"""
    try:
        status = queue_service.get_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/queue/result/{job_id}")
async def get_job_result(job_id: str):
    """Get result of a completed job"""
    try:
        result = queue_service.get_job_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found or not completed")
        return {"job_id": job_id, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job result: {str(e)}")

@app.delete("/queue/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    try:
        success = queue_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@app.get("/queue/info")
async def get_queue_info():
    """Get information about all queues"""
    try:
        queue_info = queue_service.get_queue_info()
        return {"queues": queue_info, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue info: {str(e)}")

@app.post("/queue/cleanup")
async def cleanup_old_jobs(days: int = Query(7, ge=1, le=30, description="Days to keep jobs")):
    """Clean up old completed jobs"""
    try:
        cleaned_counts = queue_service.clean_old_jobs(days)
        return {
            "message": f"Cleaned up jobs older than {days} days",
            "cleaned_counts": cleaned_counts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup jobs: {str(e)}")

@app.post("/queue/health-check")
async def queue_health_check():
    """Test queue system by running a simple health check task"""
    try:
        job_id = queue_service.enqueue_classification(health_check_task)
        return {
            "job_id": job_id,
            "message": "Health check task queued successfully",
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Failed to queue health check task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue health check: {str(e)}")

# Authentication endpoints
@app.post("/auth/register", response_model=Token)
async def register_user(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        user = auth_service.create_user(db, user_create)
        token_data = auth_service.login_user(db, UserLogin(email=user.email, password=user_create.password))
        return Token(**token_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=Token)
async def login_user(
    user_login: UserLogin,
    db: Session = Depends(get_db)
):
    """Login user and return access token"""
    try:
        token_data = auth_service.login_user(db, user_login)
        return Token(**token_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user_required)
):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

@app.get("/history", response_model=QueryHistoryResponse)
async def get_user_query_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get user's query history with pagination"""
    try:
        # Calculate offset
        offset = (page - 1) * page_size

        # Get total count
        total_count = db.query(ClassificationResult).filter(
            ClassificationResult.user_id == current_user.id
        ).count()

        # Get paginated results
        results = db.query(ClassificationResult).filter(
            ClassificationResult.user_id == current_user.id
        ).order_by(desc(ClassificationResult.created_at)).offset(offset).limit(page_size).all()

        # Convert to response format
        items = [QueryHistoryItem.from_orm(result) for result in results]

        # Calculate total pages
        total_pages = math.ceil(total_count / page_size)

        return QueryHistoryResponse(
            total_count=total_count,
            items=items,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get query history")

@app.delete("/history/{item_id}")
async def delete_query_history_item(
    item_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Delete a specific query history item"""
    try:
        # Find the item and verify ownership
        item = db.query(ClassificationResult).filter(
            ClassificationResult.id == item_id,
            ClassificationResult.user_id == current_user.id
        ).first()

        if not item:
            raise HTTPException(status_code=404, detail="Query history item not found")

        # Delete the item
        db.delete(item)
        db.commit()

        return {"message": "Query history item deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query history item: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete query history item")

@app.delete("/history")
async def delete_all_query_history(
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Delete all query history for the current user"""
    try:
        # Delete all classification results for the user
        deleted_count = db.query(ClassificationResult).filter(
            ClassificationResult.user_id == current_user.id
        ).delete()

        db.commit()

        return {
            "message": f"All query history deleted successfully",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Error deleting all query history: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete query history")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
