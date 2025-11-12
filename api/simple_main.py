"""
Simplified FastAPI backend for Vercel deployment.
This version works without heavy audio processing libraries.
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Piano Performance Analysis API",
    description="Simplified API for Vercel deployment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    message: str
    environment: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    size: int
    content_type: Optional[str] = None

class ScorePiece(BaseModel):
    id: int
    slug: str
    title: str
    created_at: str

class Performance(BaseModel):
    id: int
    score_id: int
    role: str  # 'student' or 'reference'
    source: str
    created_at: str

# Mock data for demonstration
mock_scores = [
    {"id": 1, "slug": "chopin-op9-no2", "title": "Chopin Nocturne Op. 9 No. 2", "created_at": "2024-01-01T00:00:00Z"},
    {"id": 2, "slug": "bach-wtc1-prelude1", "title": "Bach WTC Book 1 Prelude No. 1", "created_at": "2024-01-01T00:00:00Z"},
]

mock_performances = []

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Piano Performance Analysis API - Vercel Deployment",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running successfully on Vercel",
        environment=os.getenv("VERCEL_ENV", "development")
    )

# Score endpoints
@app.get("/api/scores", response_model=List[ScorePiece])
async def list_scores():
    """List all available scores."""
    return [ScorePiece(**score) for score in mock_scores]

@app.get("/api/scores/{score_id}", response_model=ScorePiece)
async def get_score(score_id: int):
    """Get a specific score by ID."""
    score = next((s for s in mock_scores if s["id"] == score_id), None)
    if not score:
        raise HTTPException(status_code=404, detail="Score not found")
    return ScorePiece(**score)

# Performance endpoints
@app.get("/api/performances", response_model=List[Performance])
async def list_performances(score_id: Optional[int] = None):
    """List performances, optionally filtered by score."""
    performances = mock_performances
    if score_id:
        performances = [p for p in performances if p["score_id"] == score_id]
    return [Performance(**perf) for perf in performances]

@app.post("/api/performances/upload", response_model=UploadResponse)
async def upload_performance(
    score_id: int = Form(...),
    role: str = Form(...),
    source: str = Form(...),
    audio: UploadFile = File(...)
):
    """Upload a performance audio file."""
    
    # Validate role
    if role not in ["student", "reference"]:
        raise HTTPException(status_code=400, detail="Role must be 'student' or 'reference'")
    
    # Validate score exists
    score = next((s for s in mock_scores if s["id"] == score_id), None)
    if not score:
        raise HTTPException(status_code=404, detail="Score not found")
    
    # Validate file
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Read file content (in real implementation, this would be processed)
    content = await audio.read()
    
    # Create mock performance record
    performance = {
        "id": len(mock_performances) + 1,
        "score_id": score_id,
        "role": role,
        "source": source,
        "created_at": "2024-01-01T00:00:00Z"
    }
    mock_performances.append(performance)
    
    logger.info(f"Uploaded {role} performance for score {score_id}: {audio.filename}")
    
    return UploadResponse(
        message=f"Successfully uploaded {role} performance",
        filename=audio.filename or "unknown",
        size=len(content),
        content_type=audio.content_type
    )

# Analysis endpoints (simplified)
@app.get("/api/analysis/{performance_id}")
async def get_analysis(performance_id: int):
    """Get analysis results for a performance."""
    performance = next((p for p in mock_performances if p["id"] == performance_id), None)
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    
    # Mock analysis results
    return {
        "performance_id": performance_id,
        "analysis": {
            "tempo": {
                "average_bpm": 120.5,
                "tempo_stability": 0.85,
                "tempo_curve": [118, 120, 122, 121, 119, 120]
            },
            "dynamics": {
                "average_loudness": -12.3,
                "dynamic_range": 18.5,
                "loudness_curve": [-15, -12, -10, -14, -16, -11]
            },
            "timing": {
                "rhythmic_accuracy": 0.92,
                "note_onset_precision": 0.88
            },
            "overall_score": 87.5
        },
        "feedback": [
            "Good tempo consistency throughout the piece",
            "Consider more dynamic contrast in the middle section",
            "Excellent rhythmic precision"
        ]
    }

# Envelope endpoints (simplified)
@app.get("/api/envelopes/{score_id}")
async def get_envelopes(score_id: int):
    """Get performance envelopes for a score."""
    score = next((s for s in mock_scores if s["id"] == score_id), None)
    if not score:
        raise HTTPException(status_code=404, detail="Score not found")
    
    # Mock envelope data
    return {
        "score_id": score_id,
        "envelopes": {
            "tempo": {
                "beats": [0, 1, 2, 3, 4, 5],
                "p20": [115, 117, 119, 118, 116, 117],
                "median": [120, 122, 124, 123, 121, 122],
                "p80": [125, 127, 129, 128, 126, 127],
                "n_refs": 3
            },
            "loudness": {
                "beats": [0, 1, 2, 3, 4, 5],
                "p20": [-18, -16, -14, -15, -17, -16],
                "median": [-12, -10, -8, -9, -11, -10],
                "p80": [-6, -4, -2, -3, -5, -4],
                "n_refs": 3
            }
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )

# Development info endpoint
@app.get("/api/info")
async def get_info():
    """Get deployment and environment information."""
    return {
        "deployment": {
            "platform": "Vercel",
            "runtime": "Python 3.9",
            "region": os.getenv("VERCEL_REGION", "unknown"),
            "env": os.getenv("VERCEL_ENV", "development")
        },
        "features": {
            "audio_upload": True,
            "basic_analysis": True,
            "mock_data": True,
            "full_audio_processing": False  # Disabled for Vercel compatibility
        },
        "endpoints": {
            "scores": "/api/scores",
            "performances": "/api/performances",
            "upload": "/api/performances/upload",
            "analysis": "/api/analysis/{performance_id}",
            "envelopes": "/api/envelopes/{score_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
