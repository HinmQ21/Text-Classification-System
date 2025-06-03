from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from datetime import datetime
import uuid

# Import database configuration
from config.database_config import engine, SessionLocal

# Base class for models
Base = declarative_base()

class ClassificationResult(Base):
    """Database model for storing classification results"""
    __tablename__ = "classification_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional for anonymous users
    text = Column(Text, nullable=False)
    model_type = Column(String(50), nullable=False)
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    language = Column(String(10), nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to user
    user = relationship("User", back_populates="classification_results")

class CSVProcessingJob(Base):
    """Database model for storing CSV processing job information"""
    __tablename__ = "csv_processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional for anonymous users
    job_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    model_type = Column(String(50), nullable=False)
    batch_size = Column(Integer, nullable=False)
    text_column = Column(String(100), nullable=False)
    total_rows = Column(Integer, nullable=False)
    processed_rows = Column(Integer, default=0)
    status = Column(String(20), default="processing")  # processing, completed, failed
    progress_percentage = Column(Float, default=0.0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="csv_jobs")
    csv_results = relationship("CSVResult", back_populates="job", cascade="all, delete-orphan")

class CSVResult(Base):
    """Database model for storing individual CSV processing results"""
    __tablename__ = "csv_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("csv_processing_jobs.job_id"), nullable=False)
    row_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    language = Column(String(10), nullable=False)
    processing_time = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to job
    job = relationship("CSVProcessingJob", back_populates="csv_results")

class User(Base):
    """Database model for users"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    classification_results = relationship("ClassificationResult", back_populates="user")
    csv_jobs = relationship("CSVProcessingJob", back_populates="user")

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_classification_result(
    db: Session,
    text: str,
    model_type: str,
    prediction: str,
    confidence: float,
    language: str,
    processing_time: float,
    user_id: int = None
) -> ClassificationResult:
    """Save classification result to database"""
    result = ClassificationResult(
        user_id=user_id,
        text=text,
        model_type=model_type,
        prediction=prediction,
        confidence=confidence,
        language=language,
        processing_time=processing_time
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result
