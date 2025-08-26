from celery import current_task
from .worker import celery_app
from .services.audio_processor import AudioProcessor
from .services.score_processor import ScoreProcessor
from .services.alignment_service import AlignmentService
from .database import SessionLocal
from .models import Analysis
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def process_analysis_task(self, analysis_id: str):
    """Background task to process analysis."""
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found")
            return
        
        # Update status to processing
        analysis.status = "processing"
        analysis.progress = 0.1
        db.commit()
        
        # Initialize processors
        audio_processor = AudioProcessor()
        score_processor = ScoreProcessor()
        alignment_service = AlignmentService(
            onset_tolerance_ms=analysis.onset_tolerance_ms,
            pitch_tolerance=analysis.pitch_tolerance
        )
        
        # Process audio
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Processing audio...'}
        )
        analysis.progress = 0.2
        db.commit()
        
        audio_result = audio_processor.transcribe_audio(analysis.audio_file_key)
        
        # Process score
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 40, 'total': 100, 'status': 'Processing score...'}
        )
        analysis.progress = 0.4
        db.commit()
        
        score_result = score_processor.process_score(analysis.score_file_key)
        
        # Convert score to reference events
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 60, 'total': 100, 'status': 'Converting score to reference...'}
        )
        analysis.progress = 0.6
        db.commit()
        
        # Estimate tempo from audio for score conversion
        tempo = audio_result.get("audio_info", {}).get("tempo", 120.0)
        score_events = score_processor.convert_to_midi_reference(score_result, tempo)
        
        # Align performance to score
        current_task.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Aligning performance to score...'}
        )
        analysis.progress = 0.8
        db.commit()
        
        alignment_result = alignment_service.align_performance_to_score(
            score_events, 
            audio_result["events"]
        )
        
        # Store results
        analysis.transcription_events = audio_result["events"]
        analysis.alignment_results = alignment_result["alignment_results"]
        analysis.accuracy_metrics = alignment_result["metrics"]
        analysis.status = "completed"
        analysis.progress = 1.0
        
        current_task.update_state(
            state='SUCCESS',
            meta={'current': 100, 'total': 100, 'status': 'Analysis completed'}
        )
        
        db.commit()
        logger.info(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {str(e)}")
        analysis.status = "failed"
        analysis.error_message = str(e)
        db.commit()
        
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise
    finally:
        db.close()


@celery_app.task
def cleanup_old_analyses():
    """Clean up old analysis files and records."""
    # This task would clean up old files and database records
    # Implementation depends on storage strategy
    pass
