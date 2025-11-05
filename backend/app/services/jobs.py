"""
Celery tasks for background audio processing.
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any
from celery import Celery
from sqlalchemy.orm import Session

from ..config import get_celery_config
from ..database import SessionLocal
from ..models import Performance, ScorePiece, RoleEnum
from ..analysis import compute_expressive_features
from .expressive import compute_and_persist_envelopes

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery('piano_analysis')
celery_app.config_from_object(get_celery_config())


@celery_app.task(bind=True, name='analyze_reference_performance')
def analyze_reference_performance(self, perf_id: int) -> Dict[str, Any]:
    """
    Analyze a reference performance in the background.
    
    Args:
        perf_id: Performance ID to analyze
        
    Returns:
        Dictionary with analysis results
    """
    db = SessionLocal()
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'step': 'Loading performance'})
        
        # Get performance record
        performance = db.query(Performance).filter(Performance.id == perf_id).first()
        if not performance:
            raise ValueError(f"Performance {perf_id} not found")
        
        if performance.role != RoleEnum.reference:
            raise ValueError(f"Performance {perf_id} is not a reference performance")
        
        # Get associated score
        score = db.query(ScorePiece).filter(ScorePiece.id == performance.score_id).first()
        if not score:
            raise ValueError(f"Score {performance.score_id} not found")
        
        logger.info(f"Starting analysis of reference performance {perf_id}")
        
        # Check if audio file exists
        audio_path = performance.source
        if not os.path.exists(audio_path):
            # If source is not a file path, we need to handle it differently
            # For now, assume it's a file path that should exist
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.update_state(state='PROGRESS', meta={'step': 'Computing expressive features'})
        
        # Compute expressive features
        # Note: This will use audio-only approximations if alignment is not provided
        features = compute_expressive_features(score, audio_path, alignment=None)
        
        if not features:
            raise ValueError("Failed to compute expressive features")
        
        self.update_state(state='PROGRESS', meta={'step': 'Saving features to database'})
        
        # Update performance with computed features
        performance.features_json = features.model_dump()
        db.commit()
        
        logger.info(f"Saved features for reference performance {perf_id}")
        
        self.update_state(state='PROGRESS', meta={'step': 'Rebuilding envelopes'})
        
        # Trigger envelope rebuild for the score
        try:
            envelopes = compute_and_persist_envelopes(performance.score_id, db)
            envelope_count = len(envelopes)
            logger.info(f"Rebuilt {envelope_count} envelopes for score {performance.score_id}")
        except Exception as e:
            logger.warning(f"Failed to rebuild envelopes: {e}")
            envelope_count = 0
        
        result = {
            'performance_id': perf_id,
            'score_id': performance.score_id,
            'features_computed': list(features.model_dump().keys()) if features else [],
            'envelopes_rebuilt': envelope_count,
            'status': 'completed'
        }
        
        logger.info(f"Completed analysis of reference performance {perf_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze reference performance {perf_id}: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='analyze_student_performance')
def analyze_student_performance(self, perf_id: int) -> Dict[str, Any]:
    """
    Analyze a student performance in the background.
    
    Args:
        perf_id: Performance ID to analyze
        
    Returns:
        Dictionary with analysis results
    """
    db = SessionLocal()
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'step': 'Loading performance'})
        
        # Get performance record
        performance = db.query(Performance).filter(Performance.id == perf_id).first()
        if not performance:
            raise ValueError(f"Performance {perf_id} not found")
        
        if performance.role != RoleEnum.student:
            raise ValueError(f"Performance {perf_id} is not a student performance")
        
        # Get associated score
        score = db.query(ScorePiece).filter(ScorePiece.id == performance.score_id).first()
        if not score:
            raise ValueError(f"Score {performance.score_id} not found")
        
        logger.info(f"Starting analysis of student performance {perf_id}")
        
        # Check if audio file exists
        audio_path = performance.source
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.update_state(state='PROGRESS', meta={'step': 'Computing expressive features'})
        
        # Compute expressive features
        features = compute_expressive_features(score, audio_path, alignment=None)
        
        if not features:
            raise ValueError("Failed to compute expressive features")
        
        self.update_state(state='PROGRESS', meta={'step': 'Saving features to database'})
        
        # Update performance with computed features
        performance.features_json = features.model_dump()
        db.commit()
        
        logger.info(f"Saved features for student performance {perf_id}")
        
        result = {
            'performance_id': perf_id,
            'score_id': performance.score_id,
            'features_computed': list(features.model_dump().keys()) if features else [],
            'features': features.model_dump() if features else None,
            'status': 'completed'
        }
        
        logger.info(f"Completed analysis of student performance {perf_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze student performance {perf_id}: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='process_audio_file')
def process_audio_file(self, audio_data: bytes, filename: str, score_id: int, role: str, source: str) -> Dict[str, Any]:
    """
    Process uploaded audio file and create performance record.
    
    Args:
        audio_data: Raw audio file bytes
        filename: Original filename
        score_id: Associated score ID
        role: Performance role ('reference' or 'student')
        source: Source description
        
    Returns:
        Dictionary with processing results
    """
    db = SessionLocal()
    
    try:
        self.update_state(state='PROGRESS', meta={'step': 'Saving audio file'})
        
        # Create temporary directory if it doesn't exist
        from ..config import settings
        temp_dir = settings.audio_temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save audio file to temporary location
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(filename)[1],
            dir=temp_dir
        )
        temp_file.write(audio_data)
        temp_file.close()
        
        temp_audio_path = temp_file.name
        
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Extracting audio metadata'})
            
            # Get audio metadata
            import soundfile as sf
            with sf.SoundFile(temp_audio_path) as f:
                sr = f.samplerate
                duration_s = len(f) / sr
            
            # Create performance record
            performance = Performance(
                score_id=score_id,
                role=RoleEnum(role),
                source=temp_audio_path,  # Store temp path for now
                sr=sr,
                duration_s=duration_s
            )
            
            db.add(performance)
            db.commit()
            db.refresh(performance)
            
            logger.info(f"Created performance {performance.id} for audio processing")
            
            self.update_state(state='PROGRESS', meta={'step': 'Starting feature analysis'})
            
            # Trigger appropriate analysis task
            if role == 'reference':
                analysis_task = analyze_reference_performance.delay(performance.id)
            else:
                analysis_task = analyze_student_performance.delay(performance.id)
            
            result = {
                'performance_id': performance.id,
                'analysis_task_id': analysis_task.id,
                'audio_path': temp_audio_path,
                'duration_s': duration_s,
                'sr': sr,
                'status': 'audio_processed'
            }
            
            return result
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise
        
    except Exception as e:
        logger.error(f"Failed to process audio file: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='cleanup_temp_files')
def cleanup_temp_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old temporary audio files.
    
    Args:
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        Cleanup statistics
    """
    try:
        from ..config import settings
        import time
        
        temp_dir = settings.audio_temp_dir
        if not os.path.exists(temp_dir):
            return {'cleaned_files': 0, 'status': 'no_temp_dir'}
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        total_size = 0
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.unlink(file_path)
                        cleaned_count += 1
                        total_size += file_size
                        logger.info(f"Cleaned up temp file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {filename}: {e}")
        
        result = {
            'cleaned_files': cleaned_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'status': 'completed'
        }
        
        logger.info(f"Cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")
        raise


# Periodic task to clean up old files
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-temp-files': {
        'task': 'cleanup_temp_files',
        'schedule': crontab(hour=2, minute=0),  # Run daily at 2 AM
        'args': (24,)  # Clean files older than 24 hours
    },
}


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a Celery task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = {
                'task_id': task_id,
                'state': result.state,
                'status': 'Task is waiting to be processed'
            }
        elif result.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'state': result.state,
                'current': result.info.get('step', ''),
                'status': 'Task is being processed'
            }
        elif result.state == 'SUCCESS':
            response = {
                'task_id': task_id,
                'state': result.state,
                'result': result.result,
                'status': 'Task completed successfully'
            }
        else:  # FAILURE
            response = {
                'task_id': task_id,
                'state': result.state,
                'error': str(result.info),
                'status': 'Task failed'
            }
        
        return response
        
    except Exception as e:
        return {
            'task_id': task_id,
            'state': 'ERROR',
            'error': str(e),
            'status': 'Failed to get task status'
        }
