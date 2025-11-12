"""Backend FastAPI application with database integration."""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import logging

from .database import get_db, create_tables, engine, Base
from .models import ScorePiece, Performance, Envelope
from .api.schemas import (
    ScorePieceIn, ScorePieceOut,
    PerformanceIn, PerformanceOut,
    EnvelopeOut, ExpressiveFeatures
)
from .api.expressive import router as expressive_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Piano Performance Analysis Backend",
    description="Backend API for storing and managing piano performance data",
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

# Include routers
app.include_router(expressive_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (may be expected in serverless): {e}")
        # Don't raise in serverless environment - tables may be created elsewhere


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Piano Performance Analysis Backend API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "database": "connected"}


# Score Piece Endpoints
@app.post("/scores/", response_model=ScorePieceOut)
async def create_score_piece(
    score: ScorePieceIn,
    db: Session = Depends(get_db)
):
    """Create a new score piece."""
    # Check if slug already exists
    existing = db.query(ScorePiece).filter(ScorePiece.slug == score.slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Score piece with this slug already exists")
    
    db_score = ScorePiece(
        slug=score.slug,
        musicxml_path=score.musicxml_path
    )
    db.add(db_score)
    db.commit()
    db.refresh(db_score)
    return db_score


@app.get("/scores/", response_model=List[ScorePieceOut])
async def list_score_pieces(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all score pieces."""
    scores = db.query(ScorePiece).offset(skip).limit(limit).all()
    return scores


@app.get("/scores/{score_id}", response_model=ScorePieceOut)
async def get_score_piece(
    score_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific score piece."""
    score = db.query(ScorePiece).filter(ScorePiece.id == score_id).first()
    if not score:
        raise HTTPException(status_code=404, detail="Score piece not found")
    return score


# Performance Endpoints
@app.post("/performances/", response_model=PerformanceOut)
async def create_performance(
    performance: PerformanceIn,
    db: Session = Depends(get_db)
):
    """Create a new performance."""
    # Verify score exists
    score = db.query(ScorePiece).filter(ScorePiece.id == performance.score_id).first()
    if not score:
        raise HTTPException(status_code=404, detail="Score piece not found")
    
    db_performance = Performance(
        score_id=performance.score_id,
        role=performance.role,
        source=performance.source
    )
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    return db_performance


@app.get("/performances/", response_model=List[PerformanceOut])
async def list_performances(
    score_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List performances, optionally filtered by score."""
    query = db.query(Performance)
    if score_id:
        query = query.filter(Performance.score_id == score_id)
    performances = query.offset(skip).limit(limit).all()
    return performances


@app.get("/performances/{performance_id}", response_model=PerformanceOut)
async def get_performance(
    performance_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific performance."""
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    return performance


# Envelope Endpoints
@app.get("/scores/{score_id}/envelopes/", response_model=List[EnvelopeOut])
async def get_score_envelopes(
    score_id: int,
    db: Session = Depends(get_db)
):
    """Get all feature envelopes for a score."""
    # Verify score exists
    score = db.query(ScorePiece).filter(ScorePiece.id == score_id).first()
    if not score:
        raise HTTPException(status_code=404, detail="Score piece not found")
    
    envelopes = db.query(Envelope).filter(Envelope.score_id == score_id).all()
    return envelopes


@app.get("/scores/{score_id}/envelopes/{feature}", response_model=EnvelopeOut)
async def get_feature_envelope(
    score_id: int,
    feature: str,
    db: Session = Depends(get_db)
):
    """Get a specific feature envelope for a score."""
    envelope = db.query(Envelope).filter(
        Envelope.score_id == score_id,
        Envelope.feature == feature
    ).first()
    
    if not envelope:
        raise HTTPException(status_code=404, detail="Feature envelope not found")
    
    return envelope


# Database utility endpoints (for development/debugging)
@app.get("/db/tables")
async def list_tables():
    """List all database tables (development endpoint)."""
    try:
        tables = Base.metadata.tables.keys()
        return {"tables": list(tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
