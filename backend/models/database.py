from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./text_classification.db")

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
