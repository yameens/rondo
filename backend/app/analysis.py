"""Beat-wise expressive feature extraction for piano performances."""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import librosa
import soundfile as sf
from scipy import signal, stats
from pathlib import Path

from .models import ScorePiece
from .api.schemas import FeatureCurve, ExpressiveFeatures
from .onset_detection import OnsetFrameDetector, detect_onsets_and_frames

# Import ByteDance piano transcription for superior accuracy
try:
    from .piano_transcription import (
        analyze_piano_performance,
        transcribe_piano_to_notes,
        PianoTranscriptionError
    )
    BYTEDANCE_AVAILABLE = True
    logger.info("ByteDance Piano Transcription available for analysis")
except ImportError as e:
    BYTEDANCE_AVAILABLE = False
    logger.warning(f"ByteDance not available, using fallback methods: {e}")

logger = logging.getLogger(__name__)

# Analysis parameters
DEFAULT_SR = 22050
HOP_LENGTH = 512
FRAME_TIME = HOP_LENGTH / DEFAULT_SR

# Feature extraction parameters
PEDAL_LOW_FREQ = 80.0  # Hz, for pedal detection
BALANCE_SPLIT_FREQ = 262.0  # Middle C, for hand balance
TEMPO_MEDIAN_FILTER_SIZE = 5  # beats, for tempo smoothing


def build_beat_grid(musicxml_path: str) -> Dict[str, List[float]]:
    """
    Build normalized beat grid from MusicXML file.
    
    Args:
        musicxml_path: Path to MusicXML file
        
    Returns:
        Dictionary with 'beats' (normalized positions) and 'nominal_durations' (in seconds)
    """
    try:
        # Try to use music21 if available
        import music21
        
        # Load the score
        score = music21.converter.parse(musicxml_path)
        
        # Get tempo marking (default to 120 BPM if not found)
        tempo_bpm = 120.0
        for element in score.flatten():
            if isinstance(element, music21.tempo.TempoIndication):
                if hasattr(element, 'number'):
                    tempo_bpm = float(element.number)
                break
        
        # Calculate beat duration in seconds
        beat_duration = 60.0 / tempo_bpm
        
        # Extract beat positions from measures
        beats = []
        nominal_durations = []
        
        current_time = 0.0
        for measure in score.parts[0].getElementsByClass(music21.stream.Measure):
            # Get time signature for this measure
            ts = measure.timeSignature or music21.meter.TimeSignature('4/4')
            beats_per_measure = ts.numerator
            
            # Add beats for this measure
            for beat_num in range(beats_per_measure):
                beats.append(current_time + beat_num * beat_duration)
                nominal_durations.append(beat_duration)
            
            # Move to next measure
            current_time += beats_per_measure * beat_duration
        
        # Normalize beat positions to [0, 1] range
        if beats:
            max_time = max(beats)
            normalized_beats = [b / max_time for b in beats]
        else:
            # Fallback: create a simple 4/4 grid
            normalized_beats = [0.0, 0.25, 0.5, 0.75, 1.0]
            nominal_durations = [beat_duration] * 5
        
        logger.info(f"Built beat grid with {len(normalized_beats)} beats at {tempo_bpm} BPM")
        
        return {
            "beats": normalized_beats,
            "nominal_durations": nominal_durations
        }
        
    except ImportError:
        logger.warning("music21 not available, using fallback beat grid")
        # Fallback: assume 4/4 time at 120 BPM
        beat_duration = 0.5  # 120 BPM = 0.5s per beat
        return {
            "beats": [0.0, 0.25, 0.5, 0.75, 1.0],
            "nominal_durations": [beat_duration] * 5
        }
    except Exception as e:
        logger.error(f"Failed to parse MusicXML {musicxml_path}: {e}")
        # Fallback: simple grid
        return {
            "beats": [0.0, 0.25, 0.5, 0.75, 1.0],
            "nominal_durations": [0.5] * 5
        }


def extract_loudness(audio_path: str, sr: int, beats: List[float]) -> FeatureCurve:
    """
    Extract beat-wise loudness (RMS) features.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        beats: Normalized beat positions [0, 1]
        
    Returns:
        FeatureCurve with loudness values per beat
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
        
        # Compute RMS with frame-based analysis
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        
        # Convert normalized beats to absolute time
        beat_times = [b * duration for b in beats]
        
        # Aggregate RMS per beat interval
        loudness_values = []
        for i, beat_time in enumerate(beat_times):
            if i < len(beat_times) - 1:
                # Get frames within this beat interval
                next_beat_time = beat_times[i + 1]
                mask = (frame_times >= beat_time) & (frame_times < next_beat_time)
            else:
                # Last beat: from current to end
                mask = frame_times >= beat_time
            
            if np.any(mask):
                # Use RMS of RMS values (energy-based aggregation)
                beat_rms = np.sqrt(np.mean(rms[mask] ** 2))
            else:
                # No frames in this interval
                beat_rms = 0.0
            
            loudness_values.append(float(beat_rms))
        
        return FeatureCurve(beats=beats, values=loudness_values)
        
    except Exception as e:
        logger.error(f"Failed to extract loudness from {audio_path}: {e}")
        # Return zero loudness
        return FeatureCurve(beats=beats, values=[0.0] * len(beats))


def extract_tempo(alignment: List[Tuple[Any, float]], beats: List[float]) -> FeatureCurve:
    """
    Extract beat-wise tempo from aligned note onsets.
    
    Args:
        alignment: List of (note_id, onset_time_seconds) tuples
        beats: Normalized beat positions [0, 1]
        
    Returns:
        FeatureCurve with tempo (BPM) values per beat
    """
    try:
        if not alignment or len(alignment) < 2:
            # No alignment data: return constant 120 BPM
            return FeatureCurve(beats=beats, values=[120.0] * len(beats))
        
        # Extract onset times and sort
        onset_times = sorted([onset for _, onset in alignment])
        
        # Compute inter-onset intervals (IOIs)
        iois = np.diff(onset_times)
        ioi_times = onset_times[1:]  # Time points for each IOI
        
        # Convert IOIs to instantaneous tempo (BPM)
        # Assume each IOI represents one beat (this is an approximation)
        instant_tempos = 60.0 / np.maximum(iois, 0.01)  # Avoid division by zero
        
        # Determine duration from alignment
        if onset_times:
            duration = max(onset_times[-1], 1.0)  # At least 1 second
        else:
            duration = 1.0
        
        # Convert normalized beats to absolute time
        beat_times = [b * duration for b in beats]
        
        # Interpolate tempo values to beat grid
        tempo_values = []
        for beat_time in beat_times:
            if len(ioi_times) > 0:
                # Find nearest tempo measurement
                idx = np.argmin(np.abs(np.array(ioi_times) - beat_time))
                tempo_values.append(float(instant_tempos[idx]))
            else:
                tempo_values.append(120.0)
        
        # Apply median filter for stability
        if len(tempo_values) >= TEMPO_MEDIAN_FILTER_SIZE:
            tempo_values = signal.medfilt(tempo_values, TEMPO_MEDIAN_FILTER_SIZE).tolist()
        
        return FeatureCurve(beats=beats, values=tempo_values)
        
    except Exception as e:
        logger.error(f"Failed to extract tempo from alignment: {e}")
        # Return constant tempo
        return FeatureCurve(beats=beats, values=[120.0] * len(beats))


def extract_articulation(
    alignment_with_durations: List[Tuple[Any, float, float]], 
    nominal_durations: List[float], 
    beats: List[float]
) -> FeatureCurve:
    """
    Extract beat-wise articulation (duration ratio) features.
    
    Args:
        alignment_with_durations: List of (note_id, onset_s, duration_s) tuples
        nominal_durations: Expected durations per beat (seconds)
        beats: Normalized beat positions [0, 1]
        
    Returns:
        FeatureCurve with articulation ratio values per beat
    """
    try:
        if not alignment_with_durations:
            # No alignment data: return neutral articulation (1.0)
            return FeatureCurve(beats=beats, values=[1.0] * len(beats))
        
        # Determine duration from alignment
        onset_times = [onset for _, onset, _ in alignment_with_durations]
        if onset_times:
            duration = max(onset_times) + 1.0  # Add buffer
        else:
            duration = 1.0
        
        # Convert normalized beats to absolute time
        beat_times = [b * duration for b in beats]
        
        # Compute articulation ratios per beat
        articulation_values = []
        
        for i, beat_time in enumerate(beat_times):
            if i < len(beat_times) - 1:
                next_beat_time = beat_times[i + 1]
            else:
                # Last beat: use nominal duration
                next_beat_time = beat_time + (nominal_durations[i] if i < len(nominal_durations) else 0.5)
            
            # Find notes that start within this beat interval
            beat_notes = []
            for note_id, onset, dur in alignment_with_durations:
                if beat_time <= onset < next_beat_time:
                    beat_notes.append(dur)
            
            if beat_notes:
                # Compute average played duration
                avg_played_duration = np.mean(beat_notes)
                # Get nominal duration for this beat
                nominal_dur = nominal_durations[i] if i < len(nominal_durations) else 0.5
                # Compute ratio
                ratio = avg_played_duration / max(nominal_dur, 0.01)  # Avoid division by zero
                articulation_values.append(float(ratio))
            else:
                # No notes in this beat: neutral articulation
                articulation_values.append(1.0)
        
        return FeatureCurve(beats=beats, values=articulation_values)
        
    except Exception as e:
        logger.error(f"Failed to extract articulation: {e}")
        # Return neutral articulation
        return FeatureCurve(beats=beats, values=[1.0] * len(beats))


def extract_pedal(audio_path: str, sr: int, beats: List[float]) -> FeatureCurve:
    """
    Extract beat-wise pedal usage proxy from audio.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        beats: Normalized beat positions [0, 1]
        
    Returns:
        FeatureCurve with pedal usage values (0-1) per beat
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
        
        # Compute spectral features for pedal detection
        # Method: Low-frequency energy + spectral flux tail
        
        # 1. Low-frequency energy (pedal resonance)
        stft = librosa.stft(y, hop_length=HOP_LENGTH)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
        
        # Focus on low frequencies (below PEDAL_LOW_FREQ)
        low_freq_mask = freqs <= PEDAL_LOW_FREQ
        low_freq_energy = np.sum(np.abs(stft[low_freq_mask, :]), axis=0)
        
        # 2. Spectral flux (rate of spectral change)
        mag = np.abs(stft)
        spectral_flux = np.sum(np.diff(mag, axis=1) * (np.diff(mag, axis=1) > 0), axis=0)
        spectral_flux = np.pad(spectral_flux, (1, 0), mode='edge')
        
        # 3. Combine features (weighted sum)
        pedal_signal = 0.7 * low_freq_energy + 0.3 * spectral_flux
        
        # Normalize to [0, 1]
        if np.max(pedal_signal) > 0:
            pedal_signal = pedal_signal / np.max(pedal_signal)
        
        # Frame times
        frame_times = librosa.frames_to_time(np.arange(len(pedal_signal)), sr=sr, hop_length=HOP_LENGTH)
        
        # Convert normalized beats to absolute time
        beat_times = [b * duration for b in beats]
        
        # Aggregate pedal signal per beat
        pedal_values = []
        for i, beat_time in enumerate(beat_times):
            if i < len(beat_times) - 1:
                next_beat_time = beat_times[i + 1]
                mask = (frame_times >= beat_time) & (frame_times < next_beat_time)
            else:
                mask = frame_times >= beat_time
            
            if np.any(mask):
                # Use mean pedal signal in this beat
                beat_pedal = np.mean(pedal_signal[mask])
            else:
                beat_pedal = 0.0
            
            pedal_values.append(float(beat_pedal))
        
        return FeatureCurve(beats=beats, values=pedal_values)
        
    except Exception as e:
        logger.error(f"Failed to extract pedal from {audio_path}: {e}")
        # Return zero pedal usage
        return FeatureCurve(beats=beats, values=[0.0] * len(beats))


def extract_balance(audio_path: str, sr: int, beats: List[float]) -> FeatureCurve:
    """
    Extract beat-wise left-right hand balance from audio.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        beats: Normalized beat positions [0, 1]
        
    Returns:
        FeatureCurve with balance values (0=left hand, 1=right hand) per beat
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
        
        # Use constant-Q transform for better frequency resolution
        cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH))
        cqt_freqs = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C1'))
        
        # Split into left hand (low) and right hand (high) registers
        split_idx = np.argmin(np.abs(cqt_freqs - BALANCE_SPLIT_FREQ))
        
        # Left hand: frequencies below split
        left_hand_energy = np.sum(cqt[:split_idx, :], axis=0)
        
        # Right hand: frequencies above split
        right_hand_energy = np.sum(cqt[split_idx:, :], axis=0)
        
        # Compute balance ratio: RH / (LH + RH)
        total_energy = left_hand_energy + right_hand_energy
        balance_ratio = np.divide(
            right_hand_energy, 
            total_energy, 
            out=np.zeros_like(right_hand_energy), 
            where=total_energy > 0
        )
        
        # Frame times
        frame_times = librosa.frames_to_time(np.arange(len(balance_ratio)), sr=sr, hop_length=HOP_LENGTH)
        
        # Convert normalized beats to absolute time
        beat_times = [b * duration for b in beats]
        
        # Aggregate balance per beat
        balance_values = []
        for i, beat_time in enumerate(beat_times):
            if i < len(beat_times) - 1:
                next_beat_time = beat_times[i + 1]
                mask = (frame_times >= beat_time) & (frame_times < next_beat_time)
            else:
                mask = frame_times >= beat_time
            
            if np.any(mask):
                # Use mean balance in this beat
                beat_balance = np.mean(balance_ratio[mask])
            else:
                beat_balance = 0.5  # Neutral balance
            
            balance_values.append(float(beat_balance))
        
        return FeatureCurve(beats=beats, values=balance_values)
        
    except Exception as e:
        logger.error(f"Failed to extract balance from {audio_path}: {e}")
        # Return neutral balance
        return FeatureCurve(beats=beats, values=[0.5] * len(beats))


def compute_expressive_features(
    score: ScorePiece, 
    perf_audio_path: str, 
    alignment: Optional[Dict[str, Any]] = None
) -> ExpressiveFeatures:
    """
    Compute all expressive features for a performance aligned to a score.
    
    Args:
        score: ScorePiece model with musicxml_path and beats_json
        perf_audio_path: Path to performance audio file
        alignment: Optional alignment data from note matching
        
    Returns:
        ExpressiveFeatures with all feature curves
    """
    try:
        logger.info(f"Computing expressive features for {perf_audio_path}")
        
        # 1. Build beat grid from score
        if score.beats_json and isinstance(score.beats_json, list):
            # Use pre-computed beat grid from score
            beats = score.beats_json
            nominal_durations = [0.5] * len(beats)  # Default duration
        else:
            # Build beat grid from MusicXML
            beat_grid = build_beat_grid(score.musicxml_path)
            beats = beat_grid["beats"]
            nominal_durations = beat_grid["nominal_durations"]
        
        # Ensure we have a valid beat grid
        if not beats:
            beats = [0.0, 0.25, 0.5, 0.75, 1.0]
            nominal_durations = [0.5] * 5
        
        logger.info(f"Using beat grid with {len(beats)} beats")
        
        # 2. Extract audio-based features
        loudness = extract_loudness(perf_audio_path, DEFAULT_SR, beats)
        pedal = extract_pedal(perf_audio_path, DEFAULT_SR, beats)
        balance = extract_balance(perf_audio_path, DEFAULT_SR, beats)
        
        # 3. Extract alignment-based features
        if alignment and "note_onsets" in alignment:
            # Use provided alignment data
            note_onsets = alignment["note_onsets"]  # List of (note_id, onset_time)
            tempo = extract_tempo(note_onsets, beats)
            
            # Check for duration data
            if "note_durations" in alignment:
                note_durations = alignment["note_durations"]  # List of (note_id, onset, duration)
                articulation = extract_articulation(note_durations, nominal_durations, beats)
            else:
                # No duration data: neutral articulation
                articulation = FeatureCurve(beats=beats, values=[1.0] * len(beats))
        else:
            # No alignment: use audio-only approximations
            logger.warning("No alignment data provided, using audio-only tempo estimation")
            
            # Audio-only tempo estimation using beat tracking
            try:
                y, _ = librosa.load(perf_audio_path, sr=DEFAULT_SR, mono=True)
                tempo_estimate, _ = librosa.beat.beat_track(y=y, sr=DEFAULT_SR)
                tempo_values = [float(tempo_estimate)] * len(beats)
            except Exception as e:
                logger.error(f"Failed audio-only tempo estimation: {e}")
                tempo_values = [120.0] * len(beats)
            
            tempo = FeatureCurve(beats=beats, values=tempo_values)
            articulation = FeatureCurve(beats=beats, values=[1.0] * len(beats))
        
        # 4. Validate all curves have same length
        expected_length = len(beats)
        for feature_name, curve in [
            ("loudness", loudness), ("tempo", tempo), 
            ("articulation", articulation), ("pedal", pedal), ("balance", balance)
        ]:
            if len(curve.values) != expected_length:
                logger.warning(f"{feature_name} curve length mismatch: {len(curve.values)} != {expected_length}")
                # Pad or truncate to match
                if len(curve.values) < expected_length:
                    curve.values.extend([curve.values[-1] if curve.values else 0.0] * (expected_length - len(curve.values)))
                else:
                    curve.values = curve.values[:expected_length]
        
        logger.info("Successfully computed all expressive features")
        
        return ExpressiveFeatures(
            tempo=tempo,
            loudness=loudness,
            articulation=articulation,
            pedal=pedal,
            balance=balance
        )
        
    except Exception as e:
        logger.error(f"Failed to compute expressive features: {e}")
        # Return empty/neutral features
        fallback_beats = [0.0, 0.25, 0.5, 0.75, 1.0]
        return ExpressiveFeatures(
            tempo=FeatureCurve(beats=fallback_beats, values=[120.0] * 5),
            loudness=FeatureCurve(beats=fallback_beats, values=[0.0] * 5),
            articulation=FeatureCurve(beats=fallback_beats, values=[1.0] * 5),
            pedal=FeatureCurve(beats=fallback_beats, values=[0.0] * 5),
            balance=FeatureCurve(beats=fallback_beats, values=[0.5] * 5)
        )


def analyze_audio_pair(student_path: str, reference_path: str = None) -> Dict[str, Any]:
    """
    Comprehensive analysis of student performance with optional reference comparison.
    Uses ByteDance Piano Transcription for superior chord detection and timing accuracy.

    Args:
        student_path: Path to student audio file
        reference_path: Optional path to reference audio file

    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info(f"Starting comprehensive audio analysis for {student_path}")

    try:
        # Analyze student performance with ByteDance (primary) or onset detection (fallback)
        if BYTEDANCE_AVAILABLE:
            try:
                logger.info("Using ByteDance Piano Transcription for superior accuracy")
                student_analysis = analyze_piano_performance(student_path)
                
                # Convert ByteDance format to expected format
                student_analysis_converted = {
                    'onsets': student_analysis['onsets'],
                    'note_frames': [(note['onset_s'], note['offset_s']) for note in student_analysis['notes']],
                    'tempo': student_analysis['tempo'],
                    'beats': student_analysis['beats'],
                    'notes': student_analysis['notes'],
                    'analysis_method': 'bytedance_piano_transcription',
                    'chord_analysis': student_analysis.get('chord_analysis', {}),
                    'transcription_quality': student_analysis.get('transcription_quality', {})
                }
                
            except Exception as e:
                logger.warning(f"ByteDance failed for student: {e}, using fallback")
                student_analysis_converted = detect_onsets_and_frames(student_path, method='librosa')
        else:
            logger.info("ByteDance not available, using multi-method onset detection")
            student_analysis_converted = detect_onsets_and_frames(student_path, method='librosa')
        
        logger.info(f"Student analysis: {len(student_analysis_converted['onsets'])} onsets detected using {student_analysis_converted.get('analysis_method', 'unknown')}")

        results = {
            'student': {
                'audio_path': student_path,
                'onset_analysis': student_analysis_converted,
                'tempo_bpm': student_analysis_converted['tempo'],
                'num_onsets': len(student_analysis_converted['onsets']),
                'num_notes': len(student_analysis_converted['note_frames']),
                'duration': student_analysis_converted.get('duration', len(student_analysis_converted['onsets']) * 0.5 if student_analysis_converted['onsets'] else 0),
                'analysis_method': student_analysis_converted.get('analysis_method', 'unknown'),
                'chord_analysis': student_analysis_converted.get('chord_analysis', {}),
                'transcription_quality': student_analysis_converted.get('transcription_quality', {})
            }
        }
        
        # Analyze reference if provided
        if reference_path:
            if BYTEDANCE_AVAILABLE:
                try:
                    logger.info("Using ByteDance Piano Transcription for reference analysis")
                    ref_analysis = analyze_piano_performance(reference_path)
                    
                    # Convert ByteDance format
                    reference_analysis = {
                        'onsets': ref_analysis['onsets'],
                        'note_frames': [(note['onset_s'], note['offset_s']) for note in ref_analysis['notes']],
                        'tempo': ref_analysis['tempo'],
                        'beats': ref_analysis['beats'],
                        'notes': ref_analysis['notes'],
                        'analysis_method': 'bytedance_piano_transcription',
                        'chord_analysis': ref_analysis.get('chord_analysis', {}),
                        'transcription_quality': ref_analysis.get('transcription_quality', {})
                    }
                    
                except Exception as e:
                    logger.warning(f"ByteDance failed for reference: {e}, using fallback")
                    reference_analysis = detect_onsets_and_frames(reference_path, method='librosa')
            else:
                reference_analysis = detect_onsets_and_frames(reference_path, method='librosa')
                
            logger.info(f"Reference analysis: {len(reference_analysis['onsets'])} onsets detected using {reference_analysis.get('analysis_method', 'unknown')}")
            
            results['reference'] = {
                'audio_path': reference_path,
                'onset_analysis': reference_analysis,
                'tempo_bpm': reference_analysis['tempo'],
                'num_onsets': len(reference_analysis['onsets']),
                'num_notes': len(reference_analysis['note_frames'])
            }
            
            # Compute comparison metrics
            results['comparison'] = compute_performance_comparison(
                student_analysis, reference_analysis
            )
        
        # Add overall assessment
        results['assessment'] = generate_performance_assessment(student_analysis, results.get('comparison'))
        
        logger.info("Comprehensive analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise


def compute_performance_comparison(student_analysis: Dict, reference_analysis: Dict) -> Dict[str, Any]:
    """
    Compare student and reference performances.
    
    Args:
        student_analysis: Student onset analysis results
        reference_analysis: Reference onset analysis results
        
    Returns:
        Dictionary with comparison metrics
    """
    try:
        comparison = {}
        
        # Tempo comparison
        student_tempo = student_analysis['tempo']
        reference_tempo = reference_analysis['tempo']
        tempo_diff = abs(student_tempo - reference_tempo)
        tempo_ratio = student_tempo / reference_tempo if reference_tempo > 0 else 1.0
        
        comparison['tempo'] = {
            'student_bpm': student_tempo,
            'reference_bpm': reference_tempo,
            'difference_bpm': tempo_diff,
            'ratio': tempo_ratio,
            'similarity_score': max(0, 100 - (tempo_diff / reference_tempo * 100)) if reference_tempo > 0 else 0
        }
        
        # Onset timing comparison
        student_onsets = np.array(student_analysis['onsets'])
        reference_onsets = np.array(reference_analysis['onsets'])
        
        if len(student_onsets) > 0 and len(reference_onsets) > 0:
            # Align onsets using DTW-like approach
            onset_alignment = align_onset_sequences(student_onsets, reference_onsets)
            comparison['timing'] = onset_alignment
        
        # Note count comparison
        student_notes = len(student_analysis['note_frames'])
        reference_notes = len(reference_analysis['note_frames'])
        
        comparison['note_count'] = {
            'student': student_notes,
            'reference': reference_notes,
            'difference': student_notes - reference_notes,
            'completeness_score': min(100, (student_notes / reference_notes * 100)) if reference_notes > 0 else 0
        }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error computing performance comparison: {e}")
        return {}


def align_onset_sequences(student_onsets: np.ndarray, reference_onsets: np.ndarray) -> Dict[str, Any]:
    """
    Align two onset sequences and compute timing accuracy metrics.
    
    Args:
        student_onsets: Array of student onset times
        reference_onsets: Array of reference onset times
        
    Returns:
        Dictionary with alignment and timing metrics
    """
    try:
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(student_onsets.reshape(-1, 1), reference_onsets.reshape(-1, 1))
        
        # Find best alignment using minimum distance
        min_len = min(len(student_onsets), len(reference_onsets))
        aligned_pairs = []
        timing_errors = []
        
        for i in range(min_len):
            if i < len(student_onsets) and i < len(reference_onsets):
                student_time = student_onsets[i]
                reference_time = reference_onsets[i]
                error = abs(student_time - reference_time)
                
                aligned_pairs.append((student_time, reference_time))
                timing_errors.append(error)
        
        if timing_errors:
            mean_error = np.mean(timing_errors)
            std_error = np.std(timing_errors)
            max_error = np.max(timing_errors)
            
            # Timing accuracy score (higher is better)
            accuracy_score = max(0, 100 - (mean_error * 100))  # Assume 1 second error = 100 point penalty
            
            return {
                'aligned_pairs': aligned_pairs,
                'mean_timing_error': mean_error,
                'std_timing_error': std_error,
                'max_timing_error': max_error,
                'accuracy_score': accuracy_score,
                'num_aligned': len(aligned_pairs)
            }
        
        return {'error': 'No valid alignments found'}
        
    except Exception as e:
        logger.error(f"Error aligning onset sequences: {e}")
        return {'error': str(e)}


def generate_performance_assessment(student_analysis: Dict, comparison: Dict = None) -> Dict[str, Any]:
    """
    Generate an overall assessment of the performance.
    
    Args:
        student_analysis: Student onset analysis results
        comparison: Optional comparison with reference
        
    Returns:
        Dictionary with assessment and feedback
    """
    try:
        assessment = {
            'overall_score': 0,
            'feedback': [],
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # Tempo assessment
        tempo = student_analysis['tempo']
        if 60 <= tempo <= 200:  # Reasonable piano tempo range
            assessment['strengths'].append(f"Good tempo control ({tempo:.1f} BPM)")
            tempo_score = 85
        elif tempo < 60:
            assessment['areas_for_improvement'].append("Tempo is quite slow - try to maintain a steadier pace")
            tempo_score = 60
        else:
            assessment['areas_for_improvement'].append("Tempo is very fast - focus on control and accuracy")
            tempo_score = 70
        
        # Onset detection quality
        num_onsets = len(student_analysis['onsets'])
        if num_onsets > 10:
            assessment['strengths'].append(f"Good note articulation - {num_onsets} clear onsets detected")
            onset_score = 80
        elif num_onsets > 5:
            assessment['feedback'].append("Moderate note clarity - some notes may need clearer articulation")
            onset_score = 70
        else:
            assessment['areas_for_improvement'].append("Few clear onsets detected - focus on note clarity")
            onset_score = 50
        
        # Comparison-based assessment
        comparison_score = 75  # Default
        if comparison:
            if 'tempo' in comparison:
                tempo_similarity = comparison['tempo'].get('similarity_score', 0)
                if tempo_similarity > 90:
                    assessment['strengths'].append("Excellent tempo matching with reference")
                    comparison_score += 10
                elif tempo_similarity > 70:
                    assessment['feedback'].append("Good tempo similarity to reference")
                else:
                    assessment['areas_for_improvement'].append("Work on matching the reference tempo")
                    comparison_score -= 10
            
            if 'timing' in comparison:
                timing_accuracy = comparison['timing'].get('accuracy_score', 0)
                if timing_accuracy > 85:
                    assessment['strengths'].append("Excellent timing accuracy")
                elif timing_accuracy > 70:
                    assessment['feedback'].append("Good timing overall")
                else:
                    assessment['areas_for_improvement'].append("Focus on timing precision")
        
        # Calculate overall score
        assessment['overall_score'] = np.mean([tempo_score, onset_score, comparison_score])
        
        # Add general feedback
        if assessment['overall_score'] > 85:
            assessment['feedback'].append("Excellent performance! Keep up the great work.")
        elif assessment['overall_score'] > 70:
            assessment['feedback'].append("Good performance with room for refinement.")
        else:
            assessment['feedback'].append("Keep practicing - focus on the areas for improvement.")
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error generating assessment: {e}")
        return {
            'overall_score': 50,
            'feedback': ['Unable to generate detailed assessment'],
            'error': str(e)
        }
