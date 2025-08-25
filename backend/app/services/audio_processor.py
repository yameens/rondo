import librosa
import numpy as np
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from typing import List, Dict, Any, Tuple
import tempfile
import os
from ..config import settings


class AudioProcessor:
    def __init__(self):
        self.model = Model(ICASSP_2022_MODEL_PATH)
    
    def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Process audio file and return transcription results.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription events and metadata
        """
        # Load and validate audio
        audio_info = self._validate_audio(audio_file_path)
        
        # Transcribe using Basic Pitch
        model_output, midi_data, note_events = predict(audio_file_path, self.model)
        
        # Normalize events to our schema
        events = self._normalize_events(note_events)
        
        return {
            "events": events,
            "audio_info": audio_info,
            "midi_data": midi_data,
            "model_output": model_output
        }
    
    def _validate_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Validate audio file and extract metadata."""
        # Load audio to get duration and sample rate
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        
        duration = len(y) / sr
        
        if duration > settings.max_audio_duration:
            raise ValueError(f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({settings.max_audio_duration}s)")
        
        # Normalize sample rate if needed
        if sr != settings.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=settings.sample_rate)
            sr = settings.sample_rate
        
        # Detect and trim leading silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        return {
            "original_duration": duration,
            "sample_rate": sr,
            "channels": 1,
            "trimmed_duration": len(y_trimmed) / sr
        }
    
    def _normalize_events(self, note_events: List[Tuple]) -> List[Dict[str, Any]]:
        """Convert Basic Pitch events to our standardized format."""
        events = []
        
        for event in note_events:
            pitch, onset, offset, velocity, confidence = event
            
            events.append({
                "pitch": int(pitch),
                "onset_s": float(onset),
                "offset_s": float(offset),
                "velocity": int(velocity),
                "confidence": float(confidence),
                "duration_s": float(offset - onset),
                "midi_note": int(pitch)
            })
        
        # Sort by onset time
        events.sort(key=lambda x: x["onset_s"])
        
        return events
    
    def extract_features(self, audio_file_path: str) -> Dict[str, Any]:
        """Extract additional audio features for analysis."""
        y, sr = librosa.load(audio_file_path, sr=settings.sample_rate, mono=True)
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Extract onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            "tempo": float(tempo),
            "onset_strength": onset_env.tolist(),
            "spectral_centroids": spectral_centroids.tolist(),
            "spectral_rolloff": spectral_rolloff.tolist()
        }
