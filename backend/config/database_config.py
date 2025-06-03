import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration class to handle MySQL and SQLite connections"""
    
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "mysql").lower()
        self.database_url = self._get_database_url()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _get_database_url(self) -> str:
        """Get database URL based on configuration"""
        if self.db_type == "mysql":
            # MySQL configuration
            db_host = os.getenv("MYSQL_HOST", "localhost")
            db_port = os.getenv("MYSQL_PORT", "3306")
            db_user = os.getenv("MYSQL_USER", "root")
            db_password = os.getenv("MYSQL_PASSWORD", "password")
            db_name = os.getenv("MYSQL_DATABASE", "text_classification")
            
            return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        
        else:
            # SQLite fallback (for development)
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "text_classification.db")
            return f"sqlite:///{db_path}"
    
    def _create_engine(self):
        """Create database engine with appropriate settings"""
        if self.db_type == "mysql":
            # MySQL engine with connection pooling
            return create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=20,  # Number of connections in the pool
                max_overflow=30,  # Additional connections that can be created
                pool_pre_ping=True,  # Validate connections before use
                pool_recycle=3600,  # Recycle connections every hour
                echo=False  # Set to True for SQL query logging in development
            )
        else:
            # SQLite engine
            return create_engine(
                self.database_url,
                connect_args={"check_same_thread": False}
            )
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_engine(self):
        """Close database engine"""
        self.engine.dispose()

# Global database configuration instance
db_config = DatabaseConfig()

# Compatibility exports
engine = db_config.engine
SessionLocal = db_config.SessionLocal
DATABASE_URL = db_config.database_url

logger.info(f"Database configured: {db_config.db_type.upper()}")
logger.info(f"Database URL: {DATABASE_URL.split('@')[0]}@****" if '@' in DATABASE_URL else DATABASE_URL) 