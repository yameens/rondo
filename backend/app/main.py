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
    import re
    
    # Generate slug from title if not provided
    slug = score.slug
    if not slug:
        # Create slug from title and composer
        slug = f"{score.composer.lower()}-{score.title.lower()}"
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
    
    # Check if slug already exists
    existing = db.query(ScorePiece).filter(ScorePiece.slug == slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Score piece with this slug already exists")
    
    db_score = ScorePiece(
        title=score.title,
        composer=score.composer,
        slug=slug,
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


@app.get("/scores/{score_id}/musicxml")
async def get_score_musicxml(
    score_id: int,
    db: Session = Depends(get_db)
):
    """
    Get MusicXML content for a score piece.
    Returns the MusicXML file content if available, or a generated placeholder.
    """
    from fastapi.responses import Response
    import os
    
    score = db.query(ScorePiece).filter(ScorePiece.id == score_id).first()
    if not score:
        raise HTTPException(status_code=404, detail="Score piece not found")
    
    # Try to load MusicXML from file if path exists
    if score.musicxml_path and os.path.exists(score.musicxml_path):
        try:
            with open(score.musicxml_path, 'r', encoding='utf-8') as f:
                musicxml_content = f.read()
            return Response(content=musicxml_content, media_type="application/xml")
        except Exception as e:
            logger.warning(f"Failed to read MusicXML file {score.musicxml_path}: {e}")
    
    # Generate a placeholder MusicXML based on the score's beat grid
    # This creates a visual representation of the analysis timeline
    beats_json = score.beats_json or [0.0, 0.25, 0.5, 0.75, 1.0]
    n_beats = len(beats_json)
    beats_per_measure = 4
    n_measures = (n_beats + beats_per_measure - 1) // beats_per_measure
    
    # Generate MusicXML with placeholder notes for each measure
    measures_xml = ""
    for measure_num in range(1, n_measures + 1):
        if measure_num == 1:
            # First measure includes attributes
            measures_xml += f'''
    <measure number="{measure_num}">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>-3</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>E</step><alter>-1</alter><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>'''
        else:
            # Subsequent measures - vary the notes slightly
            step = ['C', 'D', 'E', 'F', 'G', 'A', 'B'][(measure_num - 1) % 7]
            octave = 4 + ((measure_num - 1) // 7) % 2
            measures_xml += f'''
    <measure number="{measure_num}">
      <note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>'''
    
    # Build complete MusicXML document
    musicxml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work>
    <work-title>{score.title}</work-title>
  </work>
  <identification>
    <creator type="composer">{score.composer}</creator>
  </identification>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">{measures_xml}
  </part>
</score-partwise>'''
    
    return Response(content=musicxml_content, media_type="application/xml")


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
