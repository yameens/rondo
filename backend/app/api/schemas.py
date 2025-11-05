"""Pydantic schemas for API input/output validation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RoleEnum(str, Enum):
    """Performance role enumeration."""
    student = "student"
    reference = "reference"


class FeatureEnum(str, Enum):
    """Expressive feature enumeration."""
    tempo = "tempo"
    loudness = "loudness"
    articulation = "articulation"
    pedal = "pedal"
    balance = "balance"


# Score Piece Schemas
class ScorePieceIn(BaseModel):
    """Input schema for creating a score piece."""
    slug: str = Field(..., description="Unique identifier for the score piece")
    musicxml_path: str = Field(..., description="Path to the MusicXML file")


class ScorePieceOut(BaseModel):
    """Output schema for score piece."""
    id: int
    slug: str
    musicxml_path: str
    created_at: datetime

    class Config:
        from_attributes = True


# Performance Schemas
class PerformanceIn(BaseModel):
    """Input schema for creating a performance."""
    score_id: int = Field(..., description="ID of the associated score piece")
    role: RoleEnum = Field(..., description="Role of the performance (student or reference)")
    source: str = Field(..., description="Source file path or URL")


class PerformanceOut(BaseModel):
    """Output schema for performance."""
    id: int
    score_id: int
    role: RoleEnum
    source: str
    sr: Optional[int] = None
    duration_s: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Feature Curve Schema
class FeatureCurve(BaseModel):
    """Schema for a feature curve with beats and values."""
    beats: List[float] = Field(..., description="Beat positions")
    values: List[float] = Field(..., description="Feature values at each beat")


# Expressive Features Schema
class ExpressiveFeatures(BaseModel):
    """Schema for all expressive features."""
    tempo: Optional[FeatureCurve] = None
    loudness: Optional[FeatureCurve] = None
    articulation: Optional[FeatureCurve] = None
    pedal: Optional[FeatureCurve] = None
    balance: Optional[FeatureCurve] = None


# Envelope Schemas
class EnvelopeOut(BaseModel):
    """Output schema for feature envelope."""
    score_id: int
    feature: FeatureEnum
    beats: List[float] = Field(..., description="Beat positions")
    p20: List[float] = Field(..., description="20th percentile values")
    median: List[float] = Field(..., description="Median values")
    p80: List[float] = Field(..., description="80th percentile values")
    n_refs: int = Field(..., description="Number of reference performances")
    created_at: datetime

    class Config:
        from_attributes = True


# Extended Performance Schema with Features
class PerformanceWithFeatures(PerformanceOut):
    """Performance output schema including extracted features."""
    features: Optional[ExpressiveFeatures] = None


# Analysis Result Schema (for compatibility with existing API)
class AnalysisResult(BaseModel):
    """Schema for analysis results."""
    summary: Dict[str, Any]
    per_frame: Dict[str, List[float]]
    issues: List[Dict[str, Any]]
    tips: List[str]
    midi_estimates: Dict[str, Any]
    user_audio_url: Optional[str] = None
    ref_audio_url: Optional[str] = None
