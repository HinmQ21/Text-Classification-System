from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class TextClassificationRequest(BaseModel):
    """Request model for text classification"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    model_type: Literal["sentiment", "spam", "topic"] = Field(..., description="Type of classification model")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!",
                "model_type": "sentiment"
            }
        }

class TextClassificationResponse(BaseModel):
    """Response model for text classification"""
    text: str
    model_type: str
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: str
    processing_time: float
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I love this product!",
                "model_type": "sentiment",
                "prediction": "positive",
                "confidence": 0.95,
                "language": "en",
                "processing_time": 0.123,
                "timestamp": "2024-01-01T12:00:00"
            }
        }

class BatchRequest(BaseModel):
    """Request model for batch text classification"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    model_type: Literal["sentiment", "spam", "topic"] = Field(..., description="Type of classification model")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "This is terrible",
                    "It's okay, nothing special"
                ],
                "model_type": "sentiment"
            }
        }

class BatchResponse(BaseModel):
    """Response model for batch classification"""
    model_type: str
    total_processed: int
    results: List[dict]
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: dict
    timestamp: datetime

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    description: str
    languages: List[str]

class ModelsResponse(BaseModel):
    """Available models response"""
    available_models: List[ModelInfo]
