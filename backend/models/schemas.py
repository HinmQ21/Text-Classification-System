from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Literal, Union, Dict
from datetime import datetime
import uuid

class TextClassificationRequest(BaseModel):
    """Request model for text classification"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    model_type: Literal["sentiment", "spam", "topic"] = Field(..., description="Type of classification model")
    temperature: float = Field(default=1.0, ge=0.5, le=2.0, description="Temperature for softmax scaling (0.5-2.0)")
    model_selection: Union[str, List[str]] = Field(default="all", description="Which models to use: 'all', single model key, or list of model keys")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!",
                "model_type": "sentiment",
                "temperature": 1.0,
                "model_selection": "all"
            }
        }

class TextClassificationResponse(BaseModel):
    """Response model for text classification"""
    text: str
    model_type: str
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: Dict[str, float] = Field(..., description="Scores for all labels")
    temperature: float = Field(..., description="Temperature used for classification")
    language: str
    processing_time: float
    timestamp: datetime
    is_ensemble: bool = Field(..., description="Whether result is from ensemble of multiple models")
    models_used: List[str] = Field(..., description="List of model keys used for classification")
    individual_results: Optional[Dict[str, Dict]] = Field(None, description="Individual results from each model (if ensemble)")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "model_type": "sentiment",
                "prediction": "positive",
                "confidence": 0.95,
                "all_scores": {
                    "positive": 0.95,
                    "negative": 0.03,
                    "neutral": 0.02
                },
                "temperature": 1.0,
                "language": "en",
                "processing_time": 0.123,
                "timestamp": "2024-01-01T12:00:00",
                "is_ensemble": "false",
                "models_used": ["twitter-roberta"],
                "individual_results": "null"
            }
        }

class BatchRequest(BaseModel):
    """Request model for batch text classification"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    model_type: Literal["sentiment", "spam", "topic"] = Field(..., description="Type of classification model")
    temperature: float = Field(default=1.0, ge=0.5, le=2.0, description="Temperature for softmax scaling (0.5-2.0)")
    model_selection: Union[str, List[str]] = Field(default="all", description="Which models to use: 'all', single model key, or list of model keys")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "This is terrible",
                    "It's okay, nothing special"
                ],
                "model_type": "sentiment",
                "temperature": 1.0,
                "model_selection": "all"
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
    available_models: Dict[str, str] = Field(..., description="Available model variants with display names")

class ModelsResponse(BaseModel):
    """Available models response"""
    available_models: List[ModelInfo]

class CSVUploadRequest(BaseModel):
    """Request model for CSV upload configuration"""
    model_type: Literal["sentiment", "spam", "topic"] = Field(..., description="Type of classification model")
    batch_size: int = Field(default=16, description="Number of texts to process in each batch")
    text_column: str = Field(default="text", description="Name of the column containing text to classify")
    model_selection: Union[str, List[str]] = Field(default="all", description="Which models to use: 'all', single model key, or list of model keys")

    @validator('batch_size')
    def validate_batch_size(cls, v):
        allowed_sizes = [1, 4, 8, 16, 64, 128, 256]
        if v not in allowed_sizes:
            raise ValueError(f'Batch size must be one of {allowed_sizes}')
        return v

    class Config:
        json_json_schema_extra = {
            "example": {
                "model_type": "sentiment",
                "batch_size": 16,
                "text_column": "text",
                "model_selection": "all"
            }
        }

class CSVResultItem(BaseModel):
    """Individual result item for CSV processing"""
    row_index: int
    text: str
    prediction: str
    confidence: float
    language: str
    processing_time: float
    error: Optional[str] = None

class CSVBatchResponse(BaseModel):
    """Response model for CSV batch processing"""
    job_id: str
    status: Literal["processing", "completed", "failed"]
    model_type: str
    total_rows: int
    processed_rows: int
    batch_size: int
    progress_percentage: float
    results: List[CSVResultItem]
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None

    class Config:
        json_json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "model_type": "sentiment",
                "total_rows": 100,
                "processed_rows": 100,
                "batch_size": 20,
                "progress_percentage": 100.0,
                "results": [],
                "errors": [],
                "started_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:01:30",
                "processing_time": 90.5
            }
        }

class BatchProcessingStatus(BaseModel):
    """Status model for batch processing jobs"""
    job_id: str
    status: Literal["processing", "completed", "failed"]
    progress_percentage: float
    processed_rows: int
    total_rows: int
    estimated_time_remaining: Optional[float] = None
    current_batch: int
    total_batches: int

# Authentication Schemas
class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, max_length=100, description="User password")
    full_name: Optional[str] = Field(None, max_length=255, description="User full name")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "full_name": "John Doe"
            }
        }

class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }

class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class QueryHistoryItem(BaseModel):
    """Schema for individual query history item"""
    id: int
    text: str
    model_type: str
    prediction: str
    confidence: float
    language: str
    processing_time: float
    created_at: datetime

    class Config:
        from_attributes = True

class QueryHistoryResponse(BaseModel):
    """Schema for query history response"""
    total_count: int
    items: List[QueryHistoryItem]
    page: int
    page_size: int
    total_pages: int
