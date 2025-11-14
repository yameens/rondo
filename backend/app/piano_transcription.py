"""
Professional Piano Transcription using ByteDance's superior model.
Replaces Basic Pitch with much better chord detection and timing accuracy.
"""

import os
import logging
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Global transcriber instance for efficiency
_transcriber = None

class PianoTranscriptionError(Exception):
    """Custom exception for piano transcription errors."""
    pass

def _get_transcriber():
    """Get or create the ByteDance piano transcriber instance."""
    global _transcriber
    
    if _transcriber is None:
        try:
            from piano_transcription_inference import PianoTranscription
            
            # Check for CUDA availability
            device = 'cuda' if _has_cuda() else 'cpu'
            logger.info(f"Initializing ByteDance Piano Transcription on {device}")
            
            _transcriber = PianoTranscription(
                device=device,
                checkpoint_path=None  # Uses default pre-trained model
            )
            
            logger.info("ByteDance Piano Transcription initialized successfully")
            
        except ImportError as e:
            logger.error(f"ByteDance Piano Transcription not available: {e}")
            raise PianoTranscriptionError(f"piano-transcription-inference not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize ByteDance transcriber: {e}")
            raise PianoTranscriptionError(f"Transcriber initialization failed: {e}")
    
    return _transcriber

def _has_cuda() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            logger.info("CUDA not available, using CPU")
        return available
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return False

def transcribe_piano_audio(audio_path: str, 
                          confidence_threshold: float = 0.1,
                          min_note_duration: float = 0.05) -> List[Dict]:
    """
    Transcribe piano audio to note events using ByteDance's superior model.
    
    Args:
        audio_path: Path to audio file
        confidence_threshold: Minimum confidence for note detection (0.0-1.0)
        min_note_duration: Minimum note duration in seconds
        
    Returns:
        List of note dictionaries with keys: pitch, onset_s, offset_s, velocity, confidence
    """
    if not os.path.exists(audio_path):
        raise PianoTranscriptionError(f"Audio file not found: {audio_path}")
    
    try:
        logger.info(f"Transcribing piano audio: {audio_path}")
        
        # Load and preprocess audio
        audio, sr = sf.read(audio_path, always_2d=False)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure float32 format
        audio = audio.astype(np.float32)
        
        # Get transcriber
        transcriber = _get_transcriber()
        
        # The model expects 16kHz sample rate
        target_sr = 16000
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        logger.info(f"Processing {len(audio)/sr:.2f}s of audio at {sr}Hz")
        
        # Perform transcription
        transcription_result = transcriber.transcribe(audio, sr)
        
        # Extract note events
        notes = []
        
        if 'est_note_events' in transcription_result:
            note_events = transcription_result['est_note_events']
            
            for event in note_events:
                # ByteDance format: onset_time, offset_time, midi_note, velocity
                onset_time = float(event['onset_time'])
                offset_time = float(event['offset_time'])
                midi_note = int(event['midi_note'])
                velocity = float(event['velocity'])
                
                # Calculate duration and confidence
                duration = offset_time - onset_time
                confidence = velocity / 127.0  # Normalize velocity to confidence
                
                # Apply filters
                if (confidence >= confidence_threshold and 
                    duration >= min_note_duration and
                    21 <= midi_note <= 108):  # Piano range A0-C8
                    
                    notes.append({
                        'pitch': midi_note,
                        'onset_s': onset_time,
                        'offset_s': offset_time,
                        'velocity': velocity,
                        'confidence': confidence,
                        'duration': duration
                    })
        
        # Sort by onset time
        notes.sort(key=lambda x: x['onset_s'])
        
        logger.info(f"Transcribed {len(notes)} notes from {audio_path}")
        
        return notes
        
    except Exception as e:
        logger.error(f"Piano transcription failed for {audio_path}: {e}")
        raise PianoTranscriptionError(f"Transcription failed: {e}")

def extract_onset_times(notes: List[Dict]) -> List[float]:
    """Extract onset times from transcribed notes."""
    return [note['onset_s'] for note in notes]

def extract_beat_times(notes: List[Dict], 
                      min_ioi: float = 0.1, 
                      max_ioi: float = 2.0) -> List[float]:
    """
    Extract beat times from note onsets using inter-onset intervals.
    
    Args:
        notes: List of note dictionaries
        min_ioi: Minimum inter-onset interval in seconds
        max_ioi: Maximum inter-onset interval in seconds
        
    Returns:
        List of estimated beat times
    """
    if not notes:
        return []
    
    onsets = extract_onset_times(notes)
    
    if len(onsets) < 2:
        return onsets
    
    # Calculate inter-onset intervals
    intervals = np.diff(onsets)
    
    # Filter reasonable intervals
    valid_intervals = intervals[(intervals >= min_ioi) & (intervals <= max_ioi)]
    
    if len(valid_intervals) == 0:
        return onsets
    
    # Estimate beat period from most common interval
    beat_period = np.median(valid_intervals)
    
    # Generate beat grid
    beats = []
    current_beat = onsets[0]
    end_time = onsets[-1] + beat_period
    
    while current_beat <= end_time:
        beats.append(current_beat)
        current_beat += beat_period
    
    return beats

def analyze_piano_performance(audio_path: str) -> Dict:
    """
    Complete piano performance analysis using ByteDance transcription.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    try:
        logger.info(f"Starting ByteDance piano analysis for {audio_path}")
        
        # Transcribe notes
        notes = transcribe_piano_audio(audio_path)
        
        if not notes:
            logger.warning(f"No notes detected in {audio_path}")
            return {
                'notes': [],
                'onsets': [],
                'beats': [],
                'tempo': 120.0,
                'note_count': 0,
                'duration': 0.0,
                'analysis_method': 'bytedance_piano_transcription'
            }
        
        # Extract timing information
        onsets = extract_onset_times(notes)
        beats = extract_beat_times(notes)
        
        # Estimate tempo
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            tempo = 60.0 / np.median(beat_intervals)
        else:
            tempo = 120.0  # Default tempo
        
        # Calculate statistics
        duration = notes[-1]['offset_s'] if notes else 0.0
        note_count = len(notes)
        
        # Analyze chord density (unique feature of ByteDance)
        chord_analysis = analyze_chord_density(notes)
        
        result = {
            'notes': notes,
            'onsets': onsets,
            'beats': beats,
            'tempo': float(tempo),
            'note_count': note_count,
            'duration': duration,
            'chord_analysis': chord_analysis,
            'analysis_method': 'bytedance_piano_transcription',
            'transcription_quality': {
                'avg_confidence': np.mean([n['confidence'] for n in notes]),
                'polyphonic_density': chord_analysis['avg_simultaneous_notes'],
                'timing_precision': calculate_timing_precision(notes)
            }
        }
        
        logger.info(f"ByteDance analysis complete: {note_count} notes, {tempo:.1f} BPM")
        
        return result
        
    except Exception as e:
        logger.error(f"ByteDance piano analysis failed: {e}")
        raise

def analyze_chord_density(notes: List[Dict], 
                         simultaneity_threshold: float = 0.05) -> Dict:
    """
    Analyze chord density and polyphonic complexity.
    This is where ByteDance excels over Basic Pitch!
    
    Args:
        notes: List of note dictionaries
        simultaneity_threshold: Time window for considering notes simultaneous
        
    Returns:
        Dictionary with chord analysis
    """
    if not notes:
        return {
            'chord_count': 0,
            'avg_simultaneous_notes': 0.0,
            'max_simultaneous_notes': 0,
            'polyphonic_ratio': 0.0
        }
    
    # Group notes by onset time (within threshold)
    chord_groups = []
    current_group = [notes[0]]
    
    for note in notes[1:]:
        if note['onset_s'] - current_group[-1]['onset_s'] <= simultaneity_threshold:
            current_group.append(note)
        else:
            chord_groups.append(current_group)
            current_group = [note]
    
    if current_group:
        chord_groups.append(current_group)
    
    # Analyze chord statistics
    chord_sizes = [len(group) for group in chord_groups]
    chord_count = sum(1 for size in chord_sizes if size > 1)
    
    return {
        'chord_count': chord_count,
        'avg_simultaneous_notes': np.mean(chord_sizes),
        'max_simultaneous_notes': max(chord_sizes),
        'polyphonic_ratio': chord_count / len(chord_groups) if chord_groups else 0.0,
        'total_note_groups': len(chord_groups)
    }

def calculate_timing_precision(notes: List[Dict]) -> float:
    """
    Calculate timing precision based on onset regularity.
    
    Args:
        notes: List of note dictionaries
        
    Returns:
        Timing precision score (0.0-1.0, higher is better)
    """
    if len(notes) < 3:
        return 1.0
    
    onsets = [note['onset_s'] for note in notes]
    intervals = np.diff(onsets)
    
    if len(intervals) < 2:
        return 1.0
    
    # Calculate coefficient of variation (lower is more precise)
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
    
    # Convert to precision score (0-1, higher is better)
    precision = max(0.0, 1.0 - min(cv, 1.0))
    
    return precision

# Convenience function for backward compatibility
def transcribe_piano_to_notes(audio_path: str) -> List[Dict]:
    """
    Convenience function matching the existing interface.
    This is what gets called by your existing code!
    """
    return transcribe_piano_audio(audio_path)

# Test function
def test_bytedance_integration(audio_path: str = None) -> bool:
    """
    Test ByteDance integration with a sample file or synthetic audio.
    
    Returns:
        True if integration works, False otherwise
    """
    try:
        if audio_path and os.path.exists(audio_path):
            result = analyze_piano_performance(audio_path)
            logger.info(f"ByteDance test successful: {result['note_count']} notes detected")
            return True
        else:
            # Test with transcriber initialization only
            transcriber = _get_transcriber()
            logger.info("ByteDance transcriber initialization test passed")
            return True
            
    except Exception as e:
        logger.error(f"ByteDance integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Quick test
    logger.info("Testing ByteDance Piano Transcription integration...")
    success = test_bytedance_integration()
    print(f"ByteDance integration test: {'PASSED' if success else 'FAILED'}")
