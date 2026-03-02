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
    title: str = Field(..., description="Title of the piece")
    composer: str = Field(..., description="Composer of the piece")
    slug: Optional[str] = Field(None, description="Unique identifier (auto-generated if not provided)")
    musicxml_path: Optional[str] = Field(None, description="Path to the MusicXML file (optional)")


class ScorePieceOut(BaseModel):
    """Output schema for score piece."""
    id: int
    title: str
    composer: str
    slug: str
    musicxml_path: Optional[str] = None
    created_at: Optional[datetime] = None

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
    created_at: Optional[datetime] = None

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


# Measure-level accuracy schemas
class DeviationLevel(str, Enum):
    """Deviation level classification."""
    within = "within"
    moderate = "moderate"
    high = "high"


class MeasureScore(BaseModel):
    """Schema for per-measure accuracy data."""
    measure: int = Field(..., description="Measure number (1-indexed)")
    tempo_deviation: Optional[float] = Field(None, description="Tempo deviation for this measure")
    loudness_deviation: Optional[float] = Field(None, description="Loudness deviation for this measure")
    articulation_deviation: Optional[float] = Field(None, description="Articulation deviation for this measure")
    pedal_deviation: Optional[float] = Field(None, description="Pedal deviation for this measure")
    balance_deviation: Optional[float] = Field(None, description="Balance deviation for this measure")
    mean_deviation: float = Field(..., description="Mean deviation across all features")
    in_range: bool = Field(..., description="Whether the measure is within acceptable range")
    deviation_level: DeviationLevel = Field(..., description="Deviation classification")
    accuracy_score: float = Field(..., description="Accuracy score (0-100, higher is better)")


class ExpressiveScoreResponse(BaseModel):
    """Schema for expressive score analysis response."""
    perBeat: Dict[str, Any] = Field(..., description="Per-beat distance metrics")
    byMeasure: Dict[str, Dict[str, float]] = Field(..., description="Per-measure feature scores")
    overall: Dict[str, Any] = Field(..., description="Overall expressiveness scores")
    measure_scores: List[MeasureScore] = Field(default=[], description="Per-measure accuracy data")
