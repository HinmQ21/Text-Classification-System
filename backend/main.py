from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import logging

# Import our modules
from models.database import init_db, get_db
from models.schemas import TextClassificationRequest, TextClassificationResponse, BatchRequest
from services.text_classifier import TextClassifierService
from services.language_detector import LanguageDetectorService

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

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    init_db()
    await text_classifier.initialize()
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
        
        # Classify text
        result = await text_classifier.classify(
            text=request.text,
            model_type=request.model_type,
            language=detected_language
        )
        
        # Save to database (optional for demo)
        # save_classification_result(db, request, result)
        
        return TextClassificationResponse(
            text=request.text,
            model_type=request.model_type,
            prediction=result["prediction"],
            confidence=result["confidence"],
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
