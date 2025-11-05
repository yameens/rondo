"""
FastAPI routes for expressive analysis workflow.
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from .. import models
from . import schemas
from ..database import get_db
from ..services.expressive import (
    compute_and_persist_envelopes,
    get_envelope_for_score,
    distance_to_envelope
)
from ..analysis import build_beat_grid, compute_expressive_features
from ..services.jobs import (
    analyze_reference_performance,
    analyze_student_performance,
    process_audio_file,
    get_task_status
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["expressive"])


@router.post("/score-pieces", response_model=schemas.ScorePieceOut)
async def create_score_piece(
    score: schemas.ScorePieceIn,
    db: Session = Depends(get_db)
):
    """
    Create a new score piece and compute its beat grid.
    """
    try:
        # Build beat grid from MusicXML
        beat_data = build_beat_grid(score.musicxml_path)
        beats_json = beat_data.get("beats", [0.0, 0.25, 0.5, 0.75, 1.0])
        
        logger.info(f"Built beat grid with {len(beats_json)} beats for {score.slug}")
        
        # Create score piece
        db_score = models.ScorePiece(
            slug=score.slug,
            musicxml_path=score.musicxml_path,
            beats_json=beats_json
        )
        
        db.add(db_score)
        db.commit()
        db.refresh(db_score)
        
        logger.info(f"Created score piece {db_score.id}: {score.slug}")
        
        return db_score
        
    except Exception as e:
        logger.error(f"Failed to create score piece: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create score piece: {str(e)}")


@router.post("/performances/reference")
async def upload_reference_performance(
    score_id: int = Form(...),
    source: str = Form(...),
    audio: UploadFile = File(...),
    async_processing: bool = Query(False, alias="async"),
    db: Session = Depends(get_db)
):
    """
    Upload a reference performance audio file and compute expressive features.
    If async=1, returns job_id for background processing.
    """
    try:
        # Validate score exists
        score = db.query(models.ScorePiece).filter(models.ScorePiece.id == score_id).first()
        if not score:
            raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith(('audio/', 'application/octet-stream')):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio content
        content = await audio.read()
        
        if async_processing:
            # Process asynchronously
            logger.info(f"Starting async processing of reference audio: {audio.filename}")
            
            task = process_audio_file.delay(
                audio_data=content,
                filename=audio.filename or "audio.wav",
                score_id=score_id,
                role="reference",
                source=source
            )
            
            return {
                "job_id": task.id,
                "status": "processing",
                "message": "Audio upload received, processing in background"
            }
        
        else:
            # Process synchronously (original behavior)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(content)
                temp_audio_path = temp_file.name
            
            try:
                # Get audio info
                import soundfile as sf
                with sf.SoundFile(temp_audio_path) as f:
                    sr = f.samplerate
                    duration_s = len(f) / sr
                
                logger.info(f"Processing reference audio: {duration_s:.2f}s at {sr}Hz")
                
                # Compute expressive features
                features = compute_expressive_features(score, temp_audio_path, alignment=None)
                
                # Create performance record
                performance = models.Performance(
                    score_id=score_id,
                    role=models.RoleEnum.reference,
                    source=source,
                    sr=sr,
                    duration_s=duration_s,
                    features_json=features.model_dump() if features else None
                )
                
                db.add(performance)
                db.commit()
                db.refresh(performance)
                
                logger.info(f"Created reference performance {performance.id}")
                
                return schemas.PerformanceOut.model_validate(performance)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process reference performance: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process reference: {str(e)}")


@router.post("/performances/student")
async def upload_student_performance(
    score_id: int = Form(...),
    source: str = Form(...),
    audio: UploadFile = File(...),
    async_processing: bool = Query(False, alias="async"),
    db: Session = Depends(get_db)
):
    """
    Upload a student performance audio file and return computed features.
    If async=1, returns job_id for background processing.
    """
    try:
        # Validate score exists
        score = db.query(models.ScorePiece).filter(models.ScorePiece.id == score_id).first()
        if not score:
            raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith(('audio/', 'application/octet-stream')):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio content
        content = await audio.read()
        
        if async_processing:
            # Process asynchronously
            logger.info(f"Starting async processing of student audio: {audio.filename}")
            
            task = process_audio_file.delay(
                audio_data=content,
                filename=audio.filename or "audio.wav",
                score_id=score_id,
                role="student",
                source=source
            )
            
            return {
                "job_id": task.id,
                "status": "processing",
                "message": "Audio upload received, processing in background"
            }
        
        else:
            # Process synchronously (original behavior)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(content)
                temp_audio_path = temp_file.name
            
            try:
                # Get audio info
                import soundfile as sf
                with sf.SoundFile(temp_audio_path) as f:
                    sr = f.samplerate
                    duration_s = len(f) / sr
                
                logger.info(f"Processing student audio: {duration_s:.2f}s at {sr}Hz")
                
                # Compute expressive features
                features = compute_expressive_features(score, temp_audio_path, alignment=None)
                
                # Create performance record
                performance = models.Performance(
                    score_id=score_id,
                    role=models.RoleEnum.student,
                    source=source,
                    sr=sr,
                    duration_s=duration_s,
                    features_json=features.model_dump() if features else None
                )
                
                db.add(performance)
                db.commit()
                db.refresh(performance)
                
                logger.info(f"Created student performance {performance.id}")
                
                # Return both performance info and computed features
                return {
                    "performance": schemas.PerformanceOut.model_validate(performance),
                    "features": features.model_dump() if features else None
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process student performance: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process student: {str(e)}")


@router.post("/envelopes/{score_id}/build", response_model=Dict[str, schemas.EnvelopeOut])
async def build_envelopes(
    score_id: int,
    db: Session = Depends(get_db)
):
    """
    Build statistical envelopes from all reference performances for a score.
    """
    try:
        # Validate score exists
        score = db.query(models.ScorePiece).filter(models.ScorePiece.id == score_id).first()
        if not score:
            raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
        
        # Check if there are reference performances
        ref_count = db.query(models.Performance).filter(
            models.Performance.score_id == score_id,
            models.Performance.role == models.RoleEnum.reference,
            models.Performance.features_json.isnot(None)
        ).count()
        
        if ref_count == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No reference performances with features found for score {score_id}"
            )
        
        logger.info(f"Building envelopes from {ref_count} reference performances")
        
        # Compute and persist envelopes
        envelopes = compute_and_persist_envelopes(score_id, db)
        
        logger.info(f"Built {len(envelopes)} envelopes for score {score_id}")
        
        return envelopes
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build envelopes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build envelopes: {str(e)}")


@router.get("/envelopes/{score_id}", response_model=Dict[str, schemas.EnvelopeOut])
async def get_envelopes(
    score_id: int,
    db: Session = Depends(get_db)
):
    """
    Get existing envelopes for a score.
    """
    try:
        # Validate score exists
        score = db.query(models.ScorePiece).filter(models.ScorePiece.id == score_id).first()
        if not score:
            raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
        
        # Get all envelopes for this score
        envelopes_db = db.query(models.Envelope).filter(
            models.Envelope.score_id == score_id
        ).all()
        
        if not envelopes_db:
            raise HTTPException(status_code=404, detail=f"No envelopes found for score {score_id}")
        
        # Convert to response format
        envelopes = {}
        for envelope_db in envelopes_db:
            envelope_out = schemas.EnvelopeOut(
                score_id=envelope_db.score_id,
                feature=envelope_db.feature,
                beats=envelope_db.beats,
                p20=envelope_db.p20,
                median=envelope_db.median,
                p80=envelope_db.p80,
                n_refs=envelope_db.n_refs,
                created_at=envelope_db.created_at
            )
            envelopes[envelope_db.feature.value] = envelope_out
        
        logger.info(f"Retrieved {len(envelopes)} envelopes for score {score_id}")
        
        return envelopes
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get envelopes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get envelopes: {str(e)}")


@router.post("/expressive-score/{student_perf_id}", response_model=Dict)
async def compute_expressive_score(
    student_perf_id: int,
    db: Session = Depends(get_db)
):
    """
    Compute expressive score for a student performance against reference envelopes.
    """
    try:
        # Get student performance
        student = db.query(models.Performance).filter(
            models.Performance.id == student_perf_id,
            models.Performance.role == models.RoleEnum.student
        ).first()
        
        if not student:
            raise HTTPException(status_code=404, detail=f"Student performance {student_perf_id} not found")
        
        if not student.features_json:
            raise HTTPException(status_code=400, detail="Student performance has no computed features")
        
        # Get envelopes for the score
        envelopes_db = db.query(models.Envelope).filter(
            models.Envelope.score_id == student.score_id
        ).all()
        
        if not envelopes_db:
            raise HTTPException(
                status_code=404, 
                detail=f"No envelopes found for score {student.score_id}. Build envelopes first."
            )
        
        # Convert to EnvelopeOut objects
        envelopes = {}
        for envelope_db in envelopes_db:
            envelope_out = schemas.EnvelopeOut(
                score_id=envelope_db.score_id,
                feature=envelope_db.feature,
                beats=envelope_db.beats,
                p20=envelope_db.p20,
                median=envelope_db.median,
                p80=envelope_db.p80,
                n_refs=envelope_db.n_refs,
                created_at=envelope_db.created_at
            )
            envelopes[envelope_db.feature.value] = envelope_out
        
        logger.info(f"Computing expressive score using {len(envelopes)} envelopes")
        
        # Compute distances for each feature
        feature_distances = {}
        feature_scores = {}
        
        # Feature weights for overall score
        FEATURE_WEIGHTS = {
            'tempo': 0.40,
            'loudness': 0.40,
            'pedal': 0.10,
            'balance': 0.10,
            'articulation': 0.0  # Not included in overall score for now
        }
        
        for feature_name, envelope in envelopes.items():
            try:
                distances = distance_to_envelope(student, envelope, db)
                feature_distances[feature_name] = distances
                
                # Compute feature score (0-100, higher is better)
                # Use inverse of mean deviation, scaled and clamped
                mean_dev = distances['mean_deviation']
                # Convert to score: lower deviation = higher score
                feature_score = max(0, min(100, 100 - (mean_dev * 20)))
                feature_scores[feature_name] = feature_score
                
                logger.info(f"{feature_name}: mean_dev={mean_dev:.3f}, score={feature_score:.1f}")
                
            except Exception as e:
                logger.warning(f"Failed to compute distance for {feature_name}: {e}")
                feature_distances[feature_name] = None
                feature_scores[feature_name] = 0
        
        # Compute overall expressiveness score
        weighted_score = 0
        total_weight = 0
        
        for feature_name, weight in FEATURE_WEIGHTS.items():
            if feature_name in feature_scores and feature_scores[feature_name] is not None:
                weighted_score += feature_scores[feature_name] * weight
                total_weight += weight
        
        overall_expressiveness = weighted_score / total_weight if total_weight > 0 else 0
        
        # Group beats into measures (assuming 4 beats per measure for now)
        beats_per_measure = 4
        by_measure = {}
        
        if envelopes:
            first_envelope = next(iter(envelopes.values()))
            n_beats = len(first_envelope.beats)
            n_measures = (n_beats + beats_per_measure - 1) // beats_per_measure
            
            for measure_idx in range(n_measures):
                start_beat = measure_idx * beats_per_measure
                end_beat = min(start_beat + beats_per_measure, n_beats)
                
                measure_scores = {}
                for feature_name, distances in feature_distances.items():
                    if distances and 'scaled_deviations' in distances:
                        measure_devs = distances['scaled_deviations'][start_beat:end_beat]
                        if measure_devs:
                            avg_dev = sum(measure_devs) / len(measure_devs)
                            measure_score = max(0, min(100, 100 - (avg_dev * 20)))
                            measure_scores[feature_name] = measure_score
                
                by_measure[f"measure_{measure_idx + 1}"] = measure_scores
        
        result = {
            "perBeat": feature_distances,
            "byMeasure": by_measure,
            "overall": {
                "expressiveness": round(overall_expressiveness, 2),
                "featureScores": feature_scores,
                "weights": FEATURE_WEIGHTS
            }
        }
        
        logger.info(f"Computed expressive score: {overall_expressiveness:.2f}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compute expressive score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute expressive score: {str(e)}")


# Additional utility endpoints

@router.get("/scores/{score_id}/performances", response_model=List[schemas.PerformanceOut])
async def get_score_performances(
    score_id: int,
    role: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all performances for a score, optionally filtered by role.
    """
    try:
        query = db.query(models.Performance).filter(models.Performance.score_id == score_id)
        
        if role:
            if role not in ['student', 'reference']:
                raise HTTPException(status_code=400, detail="Role must be 'student' or 'reference'")
            query = query.filter(models.Performance.role == role)
        
        performances = query.all()
        
        return performances
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performances: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performances: {str(e)}")


@router.delete("/performances/{performance_id}")
async def delete_performance(
    performance_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a performance.
    """
    try:
        performance = db.query(models.Performance).filter(
            models.Performance.id == performance_id
        ).first()
        
        if not performance:
            raise HTTPException(status_code=404, detail=f"Performance {performance_id} not found")
        
        db.delete(performance)
        db.commit()
        
        logger.info(f"Deleted performance {performance_id}")
        
        return {"message": f"Performance {performance_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete performance: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete performance: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a background job.
    
    Returns job status: PENDING, PROGRESS, SUCCESS, or FAILURE
    """
    try:
        status_info = get_task_status(job_id)
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a background job.
    """
    try:
        from ..services.jobs import celery_app
        
        celery_app.control.revoke(job_id, terminate=True)
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancellation requested"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/jobs")
async def list_active_jobs():
    """
    List active background jobs.
    """
    try:
        from ..services.jobs import celery_app
        
        # Get active tasks
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return {"active_jobs": []}
        
        # Flatten the active tasks from all workers
        all_active = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_active.append({
                    "job_id": task["id"],
                    "name": task["name"],
                    "worker": worker,
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {})
                })
        
        return {"active_jobs": all_active}
        
    except Exception as e:
        logger.error(f"Failed to list active jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list active jobs: {str(e)}")
