"""SQLAlchemy models for piano performance analysis."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from .database import Base


class RoleEnum(str, enum.Enum):
    """Performance role enumeration."""
    student = "student"
    reference = "reference"


class FeatureEnum(str, enum.Enum):
    """Expressive feature enumeration."""
    tempo = "tempo"
    loudness = "loudness"
    articulation = "articulation"
    pedal = "pedal"
    balance = "balance"


class ScorePiece(Base):
    """Musical score piece model."""
    __tablename__ = "score_pieces"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True, nullable=False)
    musicxml_path = Column(String, nullable=False)
    beats_json = Column(JSON, nullable=True)  # Array of normalized beat indices and nominal durations
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    performances = relationship("Performance", back_populates="score")
    envelopes = relationship("Envelope", back_populates="score")


class Performance(Base):
    """Performance recording model."""
    __tablename__ = "performances"

    id = Column(Integer, primary_key=True, index=True)
    score_id = Column(Integer, ForeignKey("score_pieces.id"), nullable=False)
    role = Column(SQLEnum(RoleEnum), nullable=False)
    source = Column(String, nullable=False)  # File path or URL
    sr = Column(Integer, nullable=True)  # Sample rate
    duration_s = Column(Float, nullable=True)  # Duration in seconds
    features_json = Column(JSON, nullable=True)  # Extracted features
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    score = relationship("ScorePiece", back_populates="performances")


class Envelope(Base):
    """Expressive feature envelope model."""
    __tablename__ = "envelopes"

    id = Column(Integer, primary_key=True, index=True)
    score_id = Column(Integer, ForeignKey("score_pieces.id"), nullable=False)
    feature = Column(SQLEnum(FeatureEnum), nullable=False)
    beats = Column(JSON, nullable=False)  # Array of beat positions
    p20 = Column(JSON, nullable=False)   # 20th percentile values
    median = Column(JSON, nullable=False)  # Median values
    p80 = Column(JSON, nullable=False)   # 80th percentile values
    n_refs = Column(Integer, nullable=False, default=0)  # Number of reference performances
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    score = relationship("ScorePiece", back_populates="envelopes")

    # Unique constraint on score_id and feature
    __table_args__ = (
        Index('ix_envelope_score_feature', 'score_id', 'feature', unique=True),
    )
