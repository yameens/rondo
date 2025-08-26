export interface AnalysisData {
  id: string
  status: string
  progress: number
  results?: AlignmentResult[]
  metrics?: AccuracyMetrics
  error_message?: string
  created_at: string
  updated_at: string
}

export interface AlignmentResult {
  score_event?: NoteEvent
  performance_event?: NoteEvent
  type: string
  accuracy_type: string
  onset_delta?: number
  pitch_delta?: number
  velocity_delta?: number
  cost: number
}

export interface NoteEvent {
  pitch: number
  onset_s: number
  offset_s: number
  velocity: number
  confidence: number
  duration_s: number
  midi_note: number
  measure?: number
  part_id?: string
  type?: string
}

export interface AccuracyMetrics {
  total_score_notes: number
  total_performance_notes: number
  correct_notes: number
  missed_notes: number
  extra_notes: number
  timing_errors: number
  pitch_errors: number
  precision: number
  recall: number
  f1_score: number
  timing_rmse: number
  overall_accuracy: number
}

export interface MeasureAccuracy {
  measure_number: number
  correct_notes: number
  missed_notes: number
  extra_notes: number
  timing_errors: number
  pitch_errors: number
  total_notes: number
  accuracy: number
}

export interface UploadResponse {
  id: string
  status: string
  message: string
}

export interface AnalysisStatus {
  id: string
  status: string
  progress: number
  error_message?: string
}
