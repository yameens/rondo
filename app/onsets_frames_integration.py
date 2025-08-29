"""
Piano transcription integration using ByteDance's piano_transcription_inference.
This provides much better piano note detection than librosa's onset detection.
"""

from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def transcribe_piano_to_notes(audio_path: str) -> List[Dict]:
    """
    Transcribe a piano recording to note events using ByteDance's
    piano_transcription_inference package.
    Returns a list of dicts: {pitch, onset_s, offset_s, velocity}
    """
    try:
        from piano_transcription_inference import PianoTranscription, sample_rate
        import soundfile as sf

        # The model expects 16 kHz mono
        audio, sr = sf.read(audio_path, always_2d=False)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # mono

        transcriber = PianoTranscription(device='cuda' if _has_cuda() else 'cpu')
        # Returns a dict with 'est_note_events' (onset_s, offset_s, midi_note, velocity)
        result = transcriber.transcribe(audio, sr)

        notes = []
        for ev in result['est_note_events']:
            notes.append({
                'pitch': int(ev['midi_note']),
                'onset_s': float(ev['onset_time']),
                'offset_s': float(ev['offset_time']),
                'velocity': float(ev['velocity'] / 127.0)
            })
        # Sort by onset time
        notes.sort(key=lambda x: x['onset_s'])
        
        logger.info(f"Transcribed {len(notes)} notes from {audio_path}")
        return notes
    except ImportError as e:
        logger.warning(f"piano_transcription_inference not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Piano transcription failed for {audio_path}: {e}")
        return []


def _has_cuda() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def extract_onset_times_from_notes(notes: List[Dict]) -> np.ndarray:
    """
    Extract onset times from transcribed notes for beat tracking.
    This provides much more accurate onset detection for piano music.
    """
    if not notes:
        return np.array([])
    
    # Extract onset times and sort them
    onset_times = [note['onset_s'] for note in notes]
    onset_times = sorted(list(set(onset_times)))  # Remove duplicates
    
    return np.array(onset_times)


def extract_beat_times_from_notes(notes: List[Dict], target_bpm: float = 120.0) -> np.ndarray:
    """
    Extract beat times from transcribed notes using onset clustering.
    This is more accurate than librosa's beat tracking for piano music.
    """
    if not notes:
        return np.array([])
    
    onset_times = extract_onset_times_from_notes(notes)
    
    if len(onset_times) < 2:
        return onset_times
    
    # Use onset clustering to find beats
    # Group onsets that are close together (within 0.1 seconds)
    beat_times = []
    current_group = [onset_times[0]]
    
    for onset in onset_times[1:]:
        if onset - current_group[-1] <= 0.1:  # Within 100ms
            current_group.append(onset)
        else:
            # Take the median of the group as the beat time
            beat_times.append(np.median(current_group))
            current_group = [onset]
    
    # Handle the last group
    if current_group:
        beat_times.append(np.median(current_group))
    
    return np.array(beat_times)
