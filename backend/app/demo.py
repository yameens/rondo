from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import json
from datetime import datetime

app = FastAPI(title="Rondo Demo API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo
analyses = {}

class AnalysisCreate(BaseModel):
    id: str
    status: str
    message: str

class AnalysisStatus(BaseModel):
    id: str
    status: str
    progress: float
    error_message: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: str
    status: str
    progress: float
    results: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

@app.post("/api/v1/upload", response_model=AnalysisCreate)
async def upload_files(
    score_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    onset_tolerance_ms: Optional[int] = 50,
    pitch_tolerance: Optional[int] = 0,
):
    """Demo upload endpoint that simulates analysis."""
    
    # Validate file types
    score_ext = score_file.filename.lower().split('.')[-1] if '.' in score_file.filename else ''
    audio_ext = audio_file.filename.lower().split('.')[-1] if '.' in audio_file.filename else ''
    
    allowed_score = ['xml', 'musicxml', 'mxl', 'mei', 'pdf']
    allowed_audio = ['wav', 'mp3', 'flac']
    
    if score_ext not in allowed_score:
        raise HTTPException(status_code=400, detail=f"Unsupported score format: {score_ext}")
    
    if audio_ext not in allowed_audio:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio_ext}")
    
    # Create analysis record
    analysis_id = str(uuid.uuid4())
    analyses[analysis_id] = {
        "id": analysis_id,
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "score_filename": score_file.filename,
        "audio_filename": audio_file.filename,
        "onset_tolerance_ms": onset_tolerance_ms,
        "pitch_tolerance": pitch_tolerance
    }
    
    return AnalysisCreate(
        id=analysis_id,
        status="processing",
        message="Analysis started"
    )

@app.get("/api/v1/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Get analysis results by ID."""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[analysis_id]
    
    # Simulate completed analysis with demo data
    if analysis["status"] == "processing":
        analysis["status"] = "completed"
        analysis["progress"] = 1.0
        analysis["updated_at"] = datetime.utcnow().isoformat()
        
        # Demo results
        analysis["results"] = [
            {
                "score_event": {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5, "velocity": 80, "measure": 1},
                "performance_event": {"pitch": 60, "onset_s": 1.05, "offset_s": 1.55, "velocity": 85, "measure": 1},
                "type": "matched",
                "accuracy_type": "correct",
                "onset_delta": 0.05,
                "pitch_delta": 0,
                "velocity_delta": 5,
                "cost": 5.0
            },
            {
                "score_event": {"pitch": 64, "onset_s": 2.0, "offset_s": 2.25, "velocity": 80, "measure": 1},
                "performance_event": None,
                "type": "missed",
                "accuracy_type": "missed",
                "cost": 1000.0
            },
            {
                "score_event": None,
                "performance_event": {"pitch": 67, "onset_s": 3.0, "offset_s": 3.25, "velocity": 80, "measure": 2},
                "type": "extra",
                "accuracy_type": "extra",
                "cost": 1000.0
            }
        ]
        
        analysis["metrics"] = {
            "total_score_notes": 2,
            "total_performance_notes": 2,
            "correct_notes": 1,
            "missed_notes": 1,
            "extra_notes": 1,
            "timing_errors": 0,
            "pitch_errors": 0,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
            "timing_rmse": 0.05,
            "overall_accuracy": 0.5
        }
    
    return AnalysisResponse(**analysis)

@app.get("/api/v1/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get analysis status by ID."""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[analysis_id]
    
    return AnalysisStatus(
        id=analysis["id"],
        status=analysis["status"],
        progress=analysis["progress"],
        error_message=analysis.get("error_message")
    )

@app.get("/api/v1/analysis/{analysis_id}/export/csv")
async def export_csv(analysis_id: str):
    """Export analysis results as CSV."""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[analysis_id]
    
    if analysis["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Generate demo CSV
    csv_content = """Event_Type,Score_Pitch,Score_Onset,Score_Offset,Perf_Pitch,Perf_Onset,Perf_Offset,Accuracy_Type,Onset_Delta,Pitch_Delta,Velocity_Delta,Measure
matched,60,1.0,1.5,60,1.05,1.55,correct,0.05,0,5,1
missed,64,2.0,2.25,,,,missed,,,1
extra,,,67,3.0,3.25,extra,,,2"""
    
    return {
        "content": csv_content,
        "filename": f"rondo_analysis_{analysis_id}.csv"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rondo-demo-api"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to Rondo Demo API",
        "version": "1.0.0",
        "docs": "/docs"
    }
