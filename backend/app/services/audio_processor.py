import librosa
import numpy as np
from typing import List, Dict, Any, Tuple
from ..config import settings


class AudioProcessor:
    def __init__(self):
        """Initialize audio processor with librosa-based note detection."""
        pass
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using librosa's note detection for piano music.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription events and metadata
        """
        # Load and validate audio
        audio_info = self._validate_audio(audio_file_path)
        
        # Preprocess audio to improve transcription quality
        try:
            preprocessed_path = self._preprocess_audio(audio_file_path)
            use_preprocessed = True
        except Exception as e:
            print(f"Warning: Audio preprocessing failed: {e}. Using original audio.")
            preprocessed_path = audio_file_path
            use_preprocessed = False
        
        # Transcribe using librosa's note detection
        note_events = self._detect_notes_librosa(preprocessed_path)
        
        # Normalize events to our schema
        events = self._normalize_events(note_events)
        
        # Clean up temporary file if we created one
        if use_preprocessed and preprocessed_path != audio_file_path:
            try:
                import os
                os.unlink(preprocessed_path)
            except:
                pass
        
        return {
            "events": events,
            "audio_info": audio_info,
            "note_events": note_events
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
    
    def _detect_notes_librosa(self, audio_file_path: str) -> List[Tuple]:
        """
        Detect notes using librosa's piano-specific approach.
        
        Returns:
            List of tuples: (pitch, onset, offset, velocity, confidence)
        """
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=settings.sample_rate, mono=True)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Apply high-pass filter to remove low-frequency noise
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Detect onsets (when notes start)
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=sr,
            units='frames',
            hop_length=512,
            backtrack=True,
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=0.2,
            wait=0
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # Extract pitch using constant-Q transform (better for piano)
        C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512))
        
        # Get frequency bins
        freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=librosa.note_to_hz('A2'))
        
        # Piano frequency range (A2 to C7)
        min_freq = librosa.note_to_hz('A2')  # ~110 Hz
        max_freq = librosa.note_to_hz('C7')  # ~2093 Hz
        
        # Filter frequency range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        C_filtered = C[freq_mask]
        freqs_filtered = freqs[freq_mask]
        
        # Detect notes from onset times
        note_events = []
        
        for i, onset_time in enumerate(onset_times):
            # Get the frame index for this onset
            onset_frame = librosa.time_to_frames(onset_time, sr=sr, hop_length=512)
            
            # Look ahead a few frames to find the peak frequency
            look_ahead = min(onset_frame + 10, C_filtered.shape[1] - 1)
            
            # Find the strongest frequency at this time
            if onset_frame < C_filtered.shape[1]:
                spectrum = C_filtered[:, onset_frame:look_ahead].mean(axis=1)
                
                if len(spectrum) > 0:
                    # Find the peak frequency
                    peak_idx = np.argmax(spectrum)
                    peak_freq = freqs_filtered[peak_idx]
                    peak_amplitude = spectrum[peak_idx]
                    
                    # Convert frequency to MIDI note
                    midi_note = int(round(librosa.hz_to_midi(peak_freq)))
                    
                    # Filter out notes outside piano range
                    if 21 <= midi_note <= 108:
                        # Estimate note duration (look for next onset or end of audio)
                        if i + 1 < len(onset_times):
                            offset_time = onset_times[i + 1]
                        else:
                            # Last note - estimate duration based on amplitude decay
                            offset_time = onset_time + 2.0  # Default 2 seconds
                        
                        # Calculate velocity from amplitude (normalize to 0-127)
                        velocity = min(127, max(1, int(peak_amplitude * 100)))
                        
                        # Calculate confidence based on amplitude and frequency stability
                        confidence = min(1.0, peak_amplitude / np.max(spectrum))
                        
                        note_events.append((
                            midi_note,
                            onset_time,
                            offset_time,
                            velocity,
                            confidence
                        ))
        
        return note_events
    
    def _preprocess_audio(self, audio_file_path: str) -> str:
        """Preprocess audio to improve transcription quality."""
        import tempfile
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=settings.sample_rate, mono=True)
        
        # Apply simple audio preprocessing steps
        # 1. Normalize audio
        y = librosa.util.normalize(y)
        
        # 2. Apply high-pass filter to remove low-frequency noise
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # 3. Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # Save preprocessed audio to temporary file
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, y_trimmed, sr)
        
        return temp_path
    
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
