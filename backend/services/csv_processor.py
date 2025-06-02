import pandas as pd
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
from io import StringIO

from sqlalchemy.orm import Session
from models.database import CSVProcessingJob, CSVResult
from models.schemas import CSVUploadRequest, CSVBatchResponse, CSVResultItem, BatchProcessingStatus
from services.text_classifier import TextClassifierService
from services.language_detector import LanguageDetectorService

logger = logging.getLogger(__name__)

class CSVProcessorService:
    """Service for processing CSV files with batch text classification"""
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.text_classifier = None
        self.language_detector = None
    
    def initialize(self, text_classifier: TextClassifierService, language_detector: LanguageDetectorService):
        """Initialize with classifier and language detector services"""
        self.text_classifier = text_classifier
        self.language_detector = language_detector
    
    def validate_csv_file(self, file_content: str, text_column: str) -> Dict[str, Any]:
        """Validate CSV file format and content"""
        try:
            # Read CSV content
            df = pd.read_csv(StringIO(file_content))
            
            # Check if file is empty
            if df.empty:
                return {"valid": False, "error": "CSV file is empty"}
            
            # Check if specified text column exists
            if text_column not in df.columns:
                return {
                    "valid": False, 
                    "error": f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
                }
            
            # Check for null values in text column
            null_count = df[text_column].isnull().sum()
            if null_count > 0:
                return {
                    "valid": False,
                    "error": f"Found {null_count} empty values in '{text_column}' column"
                }
            
            # Check text length limits
            max_length = df[text_column].astype(str).str.len().max()
            if max_length > 10000:
                return {
                    "valid": False,
                    "error": f"Text too long. Maximum length is 10,000 characters, found {max_length}"
                }
            
            # Check row count limits
            if len(df) > 10000:
                return {
                    "valid": False,
                    "error": f"Too many rows. Maximum is 10,000 rows, found {len(df)}"
                }
            
            return {
                "valid": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "sample_texts": df[text_column].head(3).tolist()
            }
            
        except pd.errors.EmptyDataError:
            return {"valid": False, "error": "CSV file is empty or invalid"}
        except pd.errors.ParserError as e:
            return {"valid": False, "error": f"CSV parsing error: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": f"File validation error: {str(e)}"}
    
    async def start_csv_processing(
        self, 
        file_content: str, 
        request: CSVUploadRequest, 
        db: Session
    ) -> str:
        """Start CSV processing job and return job ID"""
        
        # Validate file first
        validation = self.validate_csv_file(file_content, request.text_column)
        if not validation["valid"]:
            raise ValueError(validation["error"])
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Read CSV data
        df = pd.read_csv(StringIO(file_content))
        total_rows = len(df)
        
        # Create job record in database
        job = CSVProcessingJob(
            job_id=job_id,
            model_type=request.model_type,
            batch_size=request.batch_size,
            text_column=request.text_column,
            total_rows=total_rows,
            status="processing"
        )
        db.add(job)
        db.commit()
        
        # Store job info in memory for progress tracking
        self.active_jobs[job_id] = {
            "status": "processing",
            "total_rows": total_rows,
            "processed_rows": 0,
            "current_batch": 0,
            "total_batches": (total_rows + request.batch_size - 1) // request.batch_size,
            "started_at": datetime.now(),
            "results": [],
            "errors": []
        }
        
        # Start processing in background
        asyncio.create_task(self._process_csv_batches(job_id, df, request, db))
        
        return job_id
    
    async def _process_csv_batches(
        self, 
        job_id: str, 
        df: pd.DataFrame, 
        request: CSVUploadRequest, 
        db: Session
    ):
        """Process CSV data in batches"""
        try:
            job_info = self.active_jobs[job_id]
            total_rows = len(df)
            batch_size = request.batch_size
            
            # Process in batches
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = df.iloc[batch_start:batch_end]
                
                job_info["current_batch"] = (batch_start // batch_size) + 1
                
                # Process each text in the batch
                for idx, row in batch_df.iterrows():
                    try:
                        text = str(row[request.text_column])
                        
                        # Detect language
                        detected_language = self.language_detector.detect(text)
                        
                        # Classify text
                        start_time = time.time()
                        result = await self.text_classifier.classify(
                            text=text,
                            model_type=request.model_type,
                            language=detected_language,
                            model_selection=getattr(request, 'model_selection', 'all')
                        )
                        processing_time = time.time() - start_time
                        
                        # Create result record
                        csv_result = CSVResult(
                            job_id=job_id,
                            row_index=idx,
                            text=text,
                            prediction=result["prediction"],
                            confidence=result["confidence"],
                            language=detected_language,
                            processing_time=processing_time
                        )
                        db.add(csv_result)
                        
                        # Add to in-memory results
                        job_info["results"].append({
                            "row_index": idx,
                            "text": text,
                            "prediction": result["prediction"],
                            "confidence": result["confidence"],
                            "language": detected_language,
                            "processing_time": processing_time
                        })
                        
                        job_info["processed_rows"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {str(e)}")
                        job_info["errors"].append(f"Row {idx}: {str(e)}")
                
                # Update progress
                progress = (job_info["processed_rows"] / total_rows) * 100
                job_info["progress_percentage"] = progress
                
                # Update database
                db.query(CSVProcessingJob).filter(
                    CSVProcessingJob.job_id == job_id
                ).update({
                    "processed_rows": job_info["processed_rows"],
                    "progress_percentage": progress
                })
                db.commit()
                
                # Small delay between batches to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Mark job as completed
            job_info["status"] = "completed"
            job_info["completed_at"] = datetime.now()
            job_info["processing_time"] = (job_info["completed_at"] - job_info["started_at"]).total_seconds()
            
            # Update database
            db.query(CSVProcessingJob).filter(
                CSVProcessingJob.job_id == job_id
            ).update({
                "status": "completed",
                "completed_at": job_info["completed_at"],
                "processing_time": job_info["processing_time"]
            })
            db.commit()
            
            logger.info(f"CSV processing job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in CSV processing job {job_id}: {str(e)}")
            
            # Mark job as failed
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["error"] = str(e)
            
            # Update database
            db.query(CSVProcessingJob).filter(
                CSVProcessingJob.job_id == job_id
            ).update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now()
            })
            db.commit()
    
    def get_job_status(self, job_id: str, db: Session) -> Optional[BatchProcessingStatus]:
        """Get current status of a processing job"""
        
        # First check in-memory cache
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            
            # Calculate estimated time remaining
            estimated_time = None
            if job_info["processed_rows"] > 0 and job_info["status"] == "processing":
                elapsed_time = (datetime.now() - job_info["started_at"]).total_seconds()
                avg_time_per_row = elapsed_time / job_info["processed_rows"]
                remaining_rows = job_info["total_rows"] - job_info["processed_rows"]
                estimated_time = avg_time_per_row * remaining_rows
            
            return BatchProcessingStatus(
                job_id=job_id,
                status=job_info["status"],
                progress_percentage=job_info.get("progress_percentage", 0.0),
                processed_rows=job_info["processed_rows"],
                total_rows=job_info["total_rows"],
                estimated_time_remaining=estimated_time,
                current_batch=job_info["current_batch"],
                total_batches=job_info["total_batches"]
            )
        
        # If not in memory, check database
        job = db.query(CSVProcessingJob).filter(CSVProcessingJob.job_id == job_id).first()
        if not job:
            return None
        
        total_batches = (job.total_rows + job.batch_size - 1) // job.batch_size
        current_batch = (job.processed_rows + job.batch_size - 1) // job.batch_size
        
        return BatchProcessingStatus(
            job_id=job_id,
            status=job.status,
            progress_percentage=job.progress_percentage,
            processed_rows=job.processed_rows,
            total_rows=job.total_rows,
            estimated_time_remaining=None,
            current_batch=current_batch,
            total_batches=total_batches
        )
    
    def get_job_results(self, job_id: str, db: Session) -> Optional[CSVBatchResponse]:
        """Get complete results for a processing job"""
        
        # Get job from database
        job = db.query(CSVProcessingJob).filter(CSVProcessingJob.job_id == job_id).first()
        if not job:
            return None
        
        # Get all results
        results = db.query(CSVResult).filter(CSVResult.job_id == job_id).all()
        
        # Convert to response format
        result_items = [
            CSVResultItem(
                row_index=result.row_index,
                text=result.text,
                prediction=result.prediction,
                confidence=result.confidence,
                language=result.language,
                processing_time=result.processing_time,
                error=result.error_message
            )
            for result in results
        ]
        
        # Get errors from in-memory cache if available
        errors = []
        if job_id in self.active_jobs:
            errors = self.active_jobs[job_id].get("errors", [])
        
        return CSVBatchResponse(
            job_id=job_id,
            status=job.status,
            model_type=job.model_type,
            total_rows=job.total_rows,
            processed_rows=job.processed_rows,
            batch_size=job.batch_size,
            progress_percentage=job.progress_percentage,
            results=result_items,
            errors=errors,
            started_at=job.started_at,
            completed_at=job.completed_at,
            processing_time=job.processing_time
        )
