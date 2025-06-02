from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import os
import uuid

# Database configuration
# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Set database path to data directory
DB_PATH = os.path.join(DATA_DIR, "text_classification.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class ClassificationResult(Base):
    """Database model for storing classification results"""
    __tablename__ = "classification_results"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    model_type = Column(String(50), nullable=False)
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    language = Column(String(10), nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class CSVProcessingJob(Base):
    """Database model for storing CSV processing job information"""
    __tablename__ = "csv_processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
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

    # Relationship to CSV results
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
    """Database model for users (for future use)"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    processing_time: float
) -> ClassificationResult:
    """Save classification result to database"""
    result = ClassificationResult(
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
