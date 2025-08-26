from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)  # For future user system
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # File references
    score_file_key = Column(String, nullable=False)
    audio_file_key = Column(String, nullable=False)
    
    # Status
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    
    # Results
    transcription_events = Column(JSON, nullable=True)
    alignment_results = Column(JSON, nullable=True)
    accuracy_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Configuration
    onset_tolerance_ms = Column(Integer, default=500)
    pitch_tolerance = Column(Integer, default=1)
    
    # Relationships
    measures = relationship("MeasureAccuracy", back_populates="analysis")


class MeasureAccuracy(Base):
    __tablename__ = "measure_accuracies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    measure_number = Column(Integer, nullable=False)
    part_id = Column(String, nullable=True)
    voice_id = Column(String, nullable=True)
    
    # Accuracy metrics
    correct_notes = Column(Integer, default=0)
    missed_notes = Column(Integer, default=0)
    extra_notes = Column(Integer, default=0)
    timing_errors = Column(Integer, default=0)
    pitch_errors = Column(Integer, default=0)
    
    # Detailed breakdown
    note_events = Column(JSON, nullable=True)  # Detailed note-by-note analysis
    
    # Relationships
    analysis = relationship("Analysis", back_populates="measures")


class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_key = Column(String, nullable=False)  # S3 key
    file_size = Column(Integer, nullable=False)
    content_type = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
