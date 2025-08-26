from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
import tempfile
import os
import uuid
from pathlib import Path

from ..database import get_db
from ..models import Analysis, FileUpload
from ..services.audio_processor import AudioProcessor
from ..services.score_processor import ScoreProcessor
from ..services.alignment_service import AlignmentService
from ..config import settings
from .schemas import AnalysisResponse, AnalysisCreate, AnalysisStatus


app = APIRouter()


@app.post("/upload", response_model=AnalysisCreate)
async def upload_files(
    score_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    onset_tolerance_ms: Optional[int] = 500,
    pitch_tolerance: Optional[int] = 1,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Upload score and audio files for analysis."""
    
    # Validate file types
    score_ext = Path(score_file.filename).suffix.lower()
    audio_ext = Path(audio_file.filename).suffix.lower()
    
    if score_ext not in settings.allowed_score_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported score format. Allowed: {settings.allowed_score_extensions}"
        )
    
    if audio_ext not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Allowed: {settings.allowed_audio_extensions}"
        )
    
    # Validate file sizes
    if score_file.size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="Score file too large")
    
    if audio_file.size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="Audio file too large")
    
    # Save files temporarily and upload to storage
    score_key = f"scores/{uuid.uuid4()}{score_ext}"
    audio_key = f"audio/{uuid.uuid4()}{audio_ext}"
    
    # For MVP, save to local filesystem
    os.makedirs("uploads", exist_ok=True)
    score_path = f"uploads/{score_key.replace('/', '_')}"
    audio_path = f"uploads/{audio_key.replace('/', '_')}"
    
    with open(score_path, "wb") as f:
        f.write(await score_file.read())
    
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Create analysis record
    analysis = Analysis(
        score_file_key=score_path,
        audio_file_key=audio_path,
        onset_tolerance_ms=onset_tolerance_ms,
        pitch_tolerance=pitch_tolerance,
        status="pending"
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Start background processing
    if background_tasks:
        background_tasks.add_task(process_analysis, analysis.id)
    
    return AnalysisCreate(
        id=analysis.id,
        status=analysis.status,
        message="Analysis started"
    )


@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Get analysis results by ID."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisResponse(
        id=analysis.id,
        status=analysis.status,
        progress=analysis.progress,
        results=analysis.alignment_results,
        metrics=analysis.accuracy_metrics,
        error_message=analysis.error_message,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at
    )


@app.get("/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str, db: Session = Depends(get_db)):
    """Get analysis status by ID."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisStatus(
        id=analysis.id,
        status=analysis.status,
        progress=analysis.progress,
        error_message=analysis.error_message
    )


@app.get("/analysis/{analysis_id}/export/csv")
async def export_csv(analysis_id: str, db: Session = Depends(get_db)):
    """Export analysis results as CSV."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis.status != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Generate CSV content
    csv_content = generate_csv_export(analysis.alignment_results)
    
    return {
        "content": csv_content,
        "filename": f"rondo_analysis_{analysis_id}.csv"
    }


async def process_analysis(analysis_id: str):
    """Background task to process analysis."""
    from ..database import SessionLocal
    
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
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
        analysis.progress = 0.2
        db.commit()
        
        audio_result = audio_processor.transcribe_audio(analysis.audio_file_key)
        
        # Process score
        analysis.progress = 0.4
        db.commit()
        
        score_result = score_processor.process_score(analysis.score_file_key)
        
        # Convert score to reference events
        analysis.progress = 0.6
        db.commit()
        
        # Estimate tempo from audio for score conversion
        tempo = audio_result.get("audio_info", {}).get("tempo", 120.0)
        score_events = score_processor.convert_to_midi_reference(score_result, tempo)
        
        # Align performance to score
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
        
        db.commit()
        
    except Exception as e:
        analysis.status = "failed"
        analysis.error_message = str(e)
        db.commit()
    finally:
        db.close()


def generate_csv_export(alignment_results: List[dict]) -> str:
    """Generate CSV export of alignment results."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "Event_Type", "Score_Pitch", "Score_Onset", "Score_Offset",
        "Perf_Pitch", "Perf_Onset", "Perf_Offset", "Accuracy_Type",
        "Onset_Delta", "Pitch_Delta", "Velocity_Delta", "Measure"
    ])
    
    # Write data
    for result in alignment_results:
        score_event = result.get("score_event", {})
        perf_event = result.get("performance_event", {})
        
        writer.writerow([
            result["type"],
            score_event.get("pitch", ""),
            score_event.get("onset_s", ""),
            score_event.get("offset_s", ""),
            perf_event.get("pitch", ""),
            perf_event.get("onset_s", ""),
            perf_event.get("offset_s", ""),
            result["accuracy_type"],
            result.get("onset_delta", ""),
            result.get("pitch_delta", ""),
            result.get("velocity_delta", ""),
            score_event.get("measure", perf_event.get("measure", ""))
        ])
    
    return output.getvalue()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rondo-api"}
