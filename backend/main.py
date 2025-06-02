from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from models.database import init_db, get_db
from models.schemas import (
    TextClassificationRequest, TextClassificationResponse, BatchRequest,
    CSVUploadRequest, CSVBatchResponse, BatchProcessingStatus
)
from services.text_classifier import TextClassifierService
from services.language_detector import LanguageDetectorService
from services.csv_processor import CSVProcessorService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="A demo API for text classification with multi-language support",
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
    return {
        "status": "healthy",
        "services": {
            "text_classifier": text_classifier.is_ready(),
            "language_detector": True,
            "database": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/classify", response_model=TextClassificationResponse)
async def classify_text(
    request: TextClassificationRequest,
    db = Depends(get_db)
):
    """
    Classify a single text input
    Supports: sentiment, spam, topic classification
    """
    try:
        # Detect language
        detected_language = language_detector.detect(request.text)
        
        # Classify text with temperature parameter
        result = await text_classifier.classify(
            text=request.text,
            model_type=request.model_type,
            language=detected_language,
            temperature=request.temperature
        )
        
        # Save to database (optional for demo)
        # save_classification_result(db, request, result)
        
        return TextClassificationResponse(
            text=request.text,
            model_type=request.model_type,
            prediction=result["prediction"],
            confidence=result["confidence"],
            all_scores=result["all_scores"],
            temperature=result["temperature"],
            language=detected_language,
            processing_time=result.get("processing_time", 0),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch")
async def classify_batch(
    request: BatchRequest,
    db = Depends(get_db)
):
    """
    Classify multiple texts in batch
    """
    try:
        results = []
        
        for text in request.texts:
            # Detect language
            detected_language = language_detector.detect(text)
            
            # Classify text
            result = await text_classifier.classify(
                text=text,
                model_type=request.model_type,
                language=detected_language
            )
            
            results.append({
                "text": text,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "language": detected_language
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

@app.get("/models")
async def get_available_models():
    """Get list of available classification models"""
    return {
        "available_models": [
            {
                "name": "sentiment",
                "description": "Sentiment Analysis (Positive/Negative/Neutral)",
                "languages": ["en", "vi", "auto"]
            },
            {
                "name": "spam",
                "description": "Spam Detection (Spam/Not Spam)",
                "languages": ["en", "auto"]
            },
            {
                "name": "topic",
                "description": "Topic Classification",
                "languages": ["en", "auto"]
            }
        ]
    }

@app.post("/classify/csv")
async def upload_csv_for_classification(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    batch_size: int = Form(default=10),
    text_column: str = Form(default="text"),
    db = Depends(get_db)
):
    """
    Upload CSV file for batch text classification
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Read file content
        file_content = await file.read()
        file_content_str = file_content.decode('utf-8')

        # Create request object
        request = CSVUploadRequest(
            model_type=model_type,
            batch_size=batch_size,
            text_column=text_column
        )

        # Start processing
        job_id = await csv_processor.start_csv_processing(file_content_str, request, db)

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
