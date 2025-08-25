from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


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
    created_at: datetime
    updated_at: datetime


class MeasureAccuracy(BaseModel):
    measure_number: int
    correct_notes: int
    missed_notes: int
    extra_notes: int
    timing_errors: int
    pitch_errors: int
    total_notes: int
    accuracy: float


class AccuracyMetrics(BaseModel):
    total_score_notes: int
    total_performance_notes: int
    correct_notes: int
    missed_notes: int
    extra_notes: int
    timing_errors: int
    pitch_errors: int
    precision: float
    recall: float
    f1_score: float
    timing_rmse: float
    overall_accuracy: float


class NoteEvent(BaseModel):
    pitch: int
    onset_s: float
    offset_s: float
    velocity: int
    confidence: float
    duration_s: float
    midi_note: int


class AlignmentResult(BaseModel):
    score_event: Optional[Dict[str, Any]]
    performance_event: Optional[Dict[str, Any]]
    type: str
    accuracy_type: str
    onset_delta: Optional[float] = None
    pitch_delta: Optional[int] = None
    velocity_delta: Optional[int] = None
    cost: float
