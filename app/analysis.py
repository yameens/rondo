#!/usr/bin/env python3
"""
Audio analysis module for comparing piano performances.
"""

import os
import csv
import logging
import numpy as np
import librosa
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# MIDI analysis imports (optional)
try:
    import basic_pitch
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False

# Piano transcription imports (new, better than Basic Pitch)
try:
    from .onsets_frames_integration import (
        transcribe_piano_to_notes,
        extract_onset_times_from_notes,
        extract_beat_times_from_notes
    )
    PIANO_TRANSCRIPTION_AVAILABLE = True
except ImportError:
    PIANO_TRANSCRIPTION_AVAILABLE = False


# Analysis parameters
TARGET_SR = 22050
HOP_LENGTH = 512
FRAME_TIME = HOP_LENGTH / TARGET_SR  # ~0.023 seconds per frame

# Thresholds for issue detection
CHROMA_DISTANCE_THRESHOLD = 0.25  # More sensitive to pitch issues
TEMPO_DIFF_THRESHOLD = 10.0  # percent
DYNAMICS_DIFF_THRESHOLD = 0.15  # normalized RMS difference
SLOPPINESS_Z_THRESHOLD = 2.0

# Minimum durations for issue detection (in resampled frames at 2 fps)
MIN_PITCH_ISSUE_FRAMES = 1  # 0.5 seconds at 2 fps
MIN_TEMPO_ISSUE_BEATS = 2  # 2 beats 
MIN_DYNAMICS_ISSUE_FRAMES = 1  # 0.5 seconds at 2 fps
MIN_SLOPPINESS_ISSUE_FRAMES = 1  # 0.5 seconds at 2 fps

# MIDI analysis parameters
MIDI_ONSET_TOLERANCE = 0.08  # seconds tolerance for onset alignment
MIDI_CONFIDENCE_THRESHOLD = 0.1  # minimum confidence for note detection (lowered to catch more notes)


def load_audio(file_path: str) -> Tuple[np.ndarray, float]:
    """
    Load audio file with standardized parameters.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (audio_data, duration_seconds)
    """
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, duration


def _extract_features_librosa(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fallback feature extraction using traditional librosa methods.
    """
    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # Beat tracking and tempo
    beats, tempo = librosa.beat.beat_track(
        onset_envelope=onset_env, 
        sr=sr, 
        hop_length=HOP_LENGTH,
        units='time'
    )
    
    return onset_env, beats, tempo


def extract_features(y: np.ndarray, sr: int = TARGET_SR, audio_path: str = None) -> Dict:
    """
    Extract comprehensive audio features for analysis.
    
    Args:
        y: Audio time series
        sr: Sample rate
        audio_path: Path to audio file (for piano transcription)
        
    Returns:
        Dictionary containing extracted features
    """
    # Try piano transcription first (much better for piano music)
    if PIANO_TRANSCRIPTION_AVAILABLE and audio_path:
        try:
            notes = transcribe_piano_to_notes(audio_path)
            if notes:
                # Use transcribed notes for better onset and beat detection
                onset_times = extract_onset_times_from_notes(notes)
                beats = extract_beat_times_from_notes(notes)
                
                # Create onset strength envelope from transcribed onsets
                duration = len(y) / sr
                frame_times = librosa.frames_to_time(np.arange(len(y) // HOP_LENGTH), sr=sr, hop_length=HOP_LENGTH)
                onset_env = np.zeros(len(frame_times))
                
                # Place onset strength at transcribed onset times
                for onset_time in onset_times:
                    frame_idx = np.argmin(np.abs(frame_times - onset_time))
                    if frame_idx < len(onset_env):
                        onset_env[frame_idx] = 1.0
                
                # Estimate tempo from beat intervals
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    tempo = 60.0 / np.median(beat_intervals)
                else:
                    tempo = 120.0  # Default tempo
                
                logger.info(f"Used piano transcription for feature extraction: {len(notes)} notes, {len(beats)} beats")
            else:
                # Fall back to librosa if transcription fails
                onset_env, beats, tempo = _extract_features_librosa(y, sr)
        except Exception as e:
            logger.warning(f"Piano transcription failed, falling back to librosa: {e}")
            onset_env, beats, tempo = _extract_features_librosa(y, sr)
    else:
        # Use traditional librosa approach
        onset_env, beats, tempo = _extract_features_librosa(y, sr)
    
    # Ensure beats is always an array
    if np.isscalar(beats):
        beats = np.array([beats])
    elif len(beats) == 0:
        # If no beats detected, create a default beat grid
        duration = len(y) / sr
        beats = np.arange(0, duration, 0.5)  # Beat every 0.5 seconds
    
    # Convert beats to frame indices for alignment
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP_LENGTH)
    
    # Chroma features (use CQT for better pitch resolution)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # Handle NaN values in chroma features
    if np.any(np.isnan(chroma)):
        logger.warning("NaN values detected in chroma features, replacing with zeros")
        chroma = np.nan_to_num(chroma, nan=0.0)
    
    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    
    # Handle NaN values in RMS
    if np.any(np.isnan(rms)):
        logger.warning("NaN values detected in RMS features, replacing with small values")
        rms = np.nan_to_num(rms, nan=1e-8)
    
    # Spectral flux (rate of change in spectrum)
    stft = librosa.stft(y, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    spectral_flux = np.sum(np.diff(mag, axis=1) * (np.diff(mag, axis=1) > 0), axis=0)
    spectral_flux = np.pad(spectral_flux, (1, 0), mode='edge')  # Pad to match frame count
    
    # Time axis for frames
    frame_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)
    
    return {
        'onset_env': onset_env,
        'beats': beats,
        'beat_frames': beat_frames,
        'tempo': tempo,
        'chroma': chroma,
        'rms': rms,
        'spectral_flux': spectral_flux,
        'frame_times': frame_times
    }


def align_with_dtw(chroma_ref: np.ndarray, chroma_user: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two chroma sequences using Dynamic Time Warping.
    
    Args:
        chroma_ref: Reference chroma features
        chroma_user: User chroma features
        
    Returns:
        Tuple of (distance_matrix, warping_path)
    """
    # Clean and validate inputs
    def clean_chroma(chroma):
        # Replace NaN with zeros
        chroma_clean = np.nan_to_num(chroma, nan=0.0)
        # Replace infinite values with zeros
        chroma_clean = np.nan_to_num(chroma_clean, posinf=0.0, neginf=0.0)
        return chroma_clean
    
    chroma_ref_clean = clean_chroma(chroma_ref)
    chroma_user_clean = clean_chroma(chroma_user)
    
    # Check for zero or near-zero chroma features
    if np.all(chroma_ref_clean == 0) or np.all(chroma_user_clean == 0):
        raise ValueError("Chroma features are all zero. Audio may be silence or too quiet.")
    
    # Normalize chroma features to prevent NaN in cosine distance
    def safe_normalize(chroma):
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        # Handle zero norms
        norms = np.where(norms < 1e-8, 1.0, norms)
        return chroma / norms
    
    chroma_ref_norm = safe_normalize(chroma_ref_clean)
    chroma_user_norm = safe_normalize(chroma_user_clean)
    
    # Final check for NaN values after normalization
    if np.any(np.isnan(chroma_ref_norm)) or np.any(np.isnan(chroma_user_norm)):
        # Fallback: use linear alignment instead of DTW
        logging.warning("NaN values detected in normalized chroma. Using linear alignment fallback.")
        min_len = min(chroma_ref_norm.shape[1], chroma_user_norm.shape[1])
        wp = np.column_stack([np.arange(min_len), np.arange(min_len)])
        D = np.zeros((min_len, min_len))  # Dummy distance matrix
        return D, wp
    
    try:
        # Compute DTW alignment with more robust parameters
        D, wp = librosa.sequence.dtw(
            X=chroma_ref_norm, 
            Y=chroma_user_norm, 
            metric='cosine',
            step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]),
            weights_add=np.array([0, 0, 0]),
            weights_mul=np.array([1, 1, 1])
        )
        
        # Reverse path for proper alignment (librosa returns path in reverse order)
        wp = wp[::-1]
        
        return D, wp
    except Exception as e:
        # Better fallback: use subsequence DTW or tempo-based resampling
        logging.warning(f"Standard DTW failed: {e}. Trying alternative alignment methods.")
        
        try:
            # Try subsequence DTW (find best matching subsequence)
            D, wp = librosa.sequence.dtw(
                X=chroma_ref_norm, 
                Y=chroma_user_norm, 
                metric='cosine',
                subseq=True
            )
            wp = wp[::-1]
            logging.info("Subsequence DTW successful")
            return D, wp
        except Exception as e2:
            logging.warning(f"Subsequence DTW also failed: {e2}. Using tempo-based resampling.")
            
            # Last resort: tempo-based resampling
            # Estimate tempo ratio and resample user to match reference length
            ref_len = chroma_ref_norm.shape[1]
            user_len = chroma_user_norm.shape[1]
            tempo_ratio = ref_len / user_len
            
            # Resample user chroma to match reference length
            user_resampled = np.zeros((12, ref_len))
            for i in range(ref_len):
                user_idx = int(i / tempo_ratio)
                user_idx = min(user_idx, user_len - 1)
                user_resampled[:, i] = chroma_user_norm[:, user_idx]
            
            # Create linear alignment with resampled data
            wp = np.column_stack([np.arange(ref_len), np.arange(ref_len)])
            D = np.zeros((ref_len, ref_len))
            
            logging.info(f"Using tempo-based resampling (ratio: {tempo_ratio:.2f})")
            return D, wp


def normalize_loudness(rms: np.ndarray) -> np.ndarray:
    """
    Normalize RMS values by integrated loudness proxy.
    
    Args:
        rms: RMS energy values
        
    Returns:
        Normalized RMS values
    """
    # Use mean RMS as a proxy for integrated loudness
    mean_rms = np.mean(rms)
    if mean_rms > 0:
        return rms / mean_rms
    return rms


def compute_tempo_curve(beats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous tempo curve from beat times.
    
    Args:
        beats: Beat times in seconds
        
    Returns:
        Tuple of (beat_times, tempo_bpm)
    """
    # Handle scalar case
    if np.isscalar(beats):
        beats = np.array([beats])
    
    if len(beats) < 2:
        return beats, np.array([120.0] * len(beats))
    
    # Compute inter-beat intervals
    ibi = np.diff(beats)
    
    # Convert to BPM (60 seconds / interval)
    tempo_bpm = 60.0 / ibi
    
    # Pad with last tempo value to match beat count
    tempo_bpm = np.pad(tempo_bpm, (0, 1), mode='edge')
    
    return beats, tempo_bpm


def resample_to_common_timebase(
    features_ref: Dict, 
    features_user: Dict, 
    wp: np.ndarray, 
    target_frame_rate: float = 2.0
) -> Dict:
    """
    Resample features to a common timebase using DTW alignment.
    
    Args:
        features_ref: Reference features
        features_user: User features
        wp: DTW warping path
        target_frame_rate: Target frames per second
        
    Returns:
        Dictionary with aligned and resampled features
    """
    # Create common time grid
    max_time = max(features_ref['frame_times'][-1], features_user['frame_times'][-1])
    common_times = np.arange(0, max_time, 1.0 / target_frame_rate)
    
    # Extract aligned frame indices
    ref_indices = wp[:, 0]
    user_indices = wp[:, 1]
    
    # Ensure indices are within bounds
    ref_indices = np.clip(ref_indices, 0, len(features_ref['frame_times']) - 1)
    user_indices = np.clip(user_indices, 0, len(features_user['frame_times']) - 1)
    
    # Get aligned features at warping path points
    ref_aligned_times = features_ref['frame_times'][ref_indices]
    user_aligned_times = features_user['frame_times'][user_indices]
    
    # Interpolate aligned features to common time grid
    def safe_interp(times, values, target_times):
        if len(times) < 2:
            return np.full_like(target_times, np.mean(values))
        # Handle edge cases where times might be out of bounds
        if len(times) == 0:
            return np.full_like(target_times, 0.0)
        # Ensure times are within bounds
        times = np.clip(times, 0, max(times))
        return np.interp(target_times, times, values)
    
    # Chroma distance at aligned points
    ref_chroma_aligned = features_ref['chroma'][:, ref_indices]
    user_chroma_aligned = features_user['chroma'][:, user_indices]
    chroma_distances = np.sqrt(np.sum((ref_chroma_aligned - user_chroma_aligned) ** 2, axis=0))
    
    # Interpolate chroma distances
    chroma_distance_interp = safe_interp(ref_aligned_times, chroma_distances, common_times)
    
    # RMS features
    ref_rms_aligned = features_ref['rms'][ref_indices]
    user_rms_aligned = features_user['rms'][user_indices]
    
    rms_ref_interp = safe_interp(ref_aligned_times, ref_rms_aligned, common_times)
    rms_user_interp = safe_interp(ref_aligned_times, user_rms_aligned, common_times)
    
    # Spectral flux
    ref_flux_aligned = features_ref['spectral_flux'][ref_indices]
    user_flux_aligned = features_user['spectral_flux'][user_indices]
    
    flux_ref_interp = safe_interp(ref_aligned_times, ref_flux_aligned, common_times)
    flux_user_interp = safe_interp(ref_aligned_times, user_flux_aligned, common_times)
    
    # Compute tempo curves for both signals
    _, tempo_ref = compute_tempo_curve(features_ref['beats'])
    _, tempo_user = compute_tempo_curve(features_user['beats'])
    
    # Interpolate tempos to common timebase
    if len(features_ref['beats']) > 1 and len(features_user['beats']) > 1:
        tempo_ref_interp = np.interp(common_times, features_ref['beats'], tempo_ref)
        tempo_user_interp = np.interp(common_times, features_user['beats'], tempo_user)
    else:
        # Fallback: use constant tempo
        tempo_ref_interp = np.full_like(common_times, 120.0)
        tempo_user_interp = np.full_like(common_times, 120.0)
    
    return {
        'time_s': common_times,
        'chroma_distance': chroma_distance_interp,
        'rms_ref': rms_ref_interp,
        'rms_user': rms_user_interp,
        'flux_ref': flux_ref_interp,
        'flux_user': flux_user_interp,
        'tempo_ref': tempo_ref_interp,
        'tempo_user': tempo_user_interp
    }


def detect_issues(aligned_features: Dict, features_ref: Dict, features_user: Dict) -> List[Dict]:
    """
    Detect performance issues based on analysis metrics.
    
    Args:
        aligned_features: Time-aligned feature dictionary
        features_ref: Reference features
        features_user: User features
        
    Returns:
        List of detected issues
    """
    issues = []
    times = aligned_features['time_s']
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.5
    
    # 1. Pitch accuracy issues (chroma distance spikes)
    chroma_high = aligned_features['chroma_distance'] > CHROMA_DISTANCE_THRESHOLD
    pitch_issue_regions = find_sustained_regions(chroma_high, MIN_PITCH_ISSUE_FRAMES)
    
    for start_idx, end_idx in pitch_issue_regions:
        start_time = times[start_idx]
        end_time = times[end_idx]
        avg_distance = np.mean(aligned_features['chroma_distance'][start_idx:end_idx+1])
        severity = min(avg_distance / CHROMA_DISTANCE_THRESHOLD, 1.0)  # Normalize to 0-1
        duration = end_time - start_time
        
        issues.append({
            'start_s': round(start_time, 2),
            'end_s': round(end_time, 2),
            'type': 'pitch_accuracy',
            'severity': round(severity, 2),
            'explanation': f'Likely pitch mismatch (notes off or extra) for {duration:.1f}s'
        })
    
    # 2. Tempo drift/surge
    _, tempo_ref = compute_tempo_curve(features_ref['beats'])
    _, tempo_user = compute_tempo_curve(features_user['beats'])
    
    # Interpolate tempos to common timebase for comparison
    if len(features_ref['beats']) > 1 and len(features_user['beats']) > 1:
        tempo_ref_interp = np.interp(times, features_ref['beats'], tempo_ref)
        tempo_user_interp = np.interp(times, features_user['beats'], tempo_user)
        
        tempo_diff_pct = 100 * np.abs(tempo_user_interp - tempo_ref_interp) / tempo_ref_interp
        tempo_high = tempo_diff_pct > TEMPO_DIFF_THRESHOLD
        tempo_issue_regions = find_sustained_regions(tempo_high, MIN_TEMPO_ISSUE_BEATS)
        
        for start_idx, end_idx in tempo_issue_regions:
            start_time = times[start_idx]
            end_time = times[end_idx]
            avg_diff = np.mean(tempo_diff_pct[start_idx:end_idx+1])
            severity = min(avg_diff / TEMPO_DIFF_THRESHOLD, 1.0)  # Normalize to 0-1
            
            # Calculate number of beats in this region (approximate)
            beats_in_region = max(1, int((end_time - start_time) * np.mean(tempo_user_interp[start_idx:end_idx+1]) / 60))
            
            # Determine if tempo is faster or slower
            avg_tempo_diff = np.mean(tempo_user_interp[start_idx:end_idx+1] - tempo_ref_interp[start_idx:end_idx+1])
            sign = "+" if avg_tempo_diff > 0 else "-"
            
            issues.append({
                'start_s': round(start_time, 2),
                'end_s': round(end_time, 2),
                'type': 'tempo_drift',
                'severity': round(severity, 2),
                'explanation': f'Played ≈ {sign}{avg_diff:.0f}% vs reference over {beats_in_region} beats'
            })
    
    # 3. Dynamics mismatch
    rms_ref_norm = normalize_loudness(aligned_features['rms_ref'])
    rms_user_norm = normalize_loudness(aligned_features['rms_user'])
    dynamics_diff = np.abs(rms_user_norm - rms_ref_norm)
    
    dynamics_high = dynamics_diff > DYNAMICS_DIFF_THRESHOLD
    dynamics_issue_regions = find_sustained_regions(dynamics_high, MIN_DYNAMICS_ISSUE_FRAMES)
    
    for start_idx, end_idx in dynamics_issue_regions:
        start_time = times[start_idx]
        end_time = times[end_idx]
        avg_diff = np.mean(dynamics_diff[start_idx:end_idx+1])
        severity = min(avg_diff / DYNAMICS_DIFF_THRESHOLD, 1.0)  # Normalize to 0-1
        duration = end_time - start_time
        
        # Determine if user is louder or softer
        user_louder = np.mean(rms_user_norm[start_idx:end_idx+1]) > np.mean(rms_ref_norm[start_idx:end_idx+1])
        loudness_desc = "louder" if user_louder else "softer"
        
        issues.append({
            'start_s': round(start_time, 2),
            'end_s': round(end_time, 2),
            'type': 'dynamics_mismatch',
            'severity': round(severity, 2),
            'explanation': f'Much {loudness_desc} than reference for {duration:.1f}s'
        })
    
    # 4. Sloppiness (spectral flux anomalies)
    if len(aligned_features['flux_user']) > 1:
        flux_z_scores = np.abs(stats.zscore(aligned_features['flux_user']))
        sloppiness_high = flux_z_scores > SLOPPINESS_Z_THRESHOLD
        sloppiness_issue_regions = find_sustained_regions(sloppiness_high, MIN_SLOPPINESS_ISSUE_FRAMES)
        
        for start_idx, end_idx in sloppiness_issue_regions:
            start_time = times[start_idx]
            end_time = times[end_idx]
            avg_z_score = np.mean(flux_z_scores[start_idx:end_idx+1])
            severity = min(avg_z_score / SLOPPINESS_Z_THRESHOLD, 1.0)  # Normalize to 0-1
            
            issues.append({
                'start_s': round(start_time, 2),
                'end_s': round(end_time, 2),
                'type': 'messy_articulation',
                'severity': round(severity, 2),
                'explanation': 'Irregular articulation; consider slower practice'
            })
    
    return issues


def find_sustained_regions(boolean_array: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """
    Find regions where boolean condition is sustained for minimum length.
    
    Args:
        boolean_array: Boolean array indicating condition
        min_length: Minimum length for sustained region
        
    Returns:
        List of (start_index, end_index) tuples
    """
    regions = []
    if len(boolean_array) == 0:
        return regions
    
    # Find transition points
    diff = np.diff(np.concatenate(([False], boolean_array, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    # Filter by minimum length
    for start, end in zip(starts, ends):
        if (end - start + 1) >= min_length:
            regions.append((start, end))
    
    return regions


def compute_scores(aligned_features: Dict, issues: List[Dict]) -> Dict[str, float]:
    """
    Compute performance scores based on analysis.
    
    Args:
        aligned_features: Time-aligned features
        issues: List of detected issues
        
    Returns:
        Dictionary of performance scores
    """
    # Pitch score (based on chroma distance)
    mean_chroma_distance = np.mean(aligned_features['chroma_distance'])
    # More reasonable pitch scoring: 0.5 distance = 50% score, 1.0 distance = 0% score
    pitch_score = max(0, 100 * (1 - mean_chroma_distance / 1.0))
    
    # Tempo score (based on actual tempo differences, not just drift issues)
    if 'tempo_user' in aligned_features and 'tempo_ref' in aligned_features:
        tempo_diff = np.mean(np.abs(aligned_features['tempo_user'] - aligned_features['tempo_ref']))
        # 20 BPM difference = 0% score, 0 BPM difference = 100% score
        tempo_score = max(0, 100 * (1 - tempo_diff / 20.0))
    else:
        # Fallback: count tempo drift issues
        tempo_penalty = sum(1 for issue in issues if issue['type'] == 'tempo_drift')
        tempo_score = max(0, 100 - 10 * tempo_penalty)
    
    # Dynamics score (based on RMS similarity)
    dynamics_diff = np.mean(np.abs(
        normalize_loudness(aligned_features['rms_user']) - 
        normalize_loudness(aligned_features['rms_ref'])
    ))
    # More reasonable dynamics scoring: 0.3 difference = 50% score, 0.6 difference = 0% score
    dynamics_score = max(0, 100 * (1 - dynamics_diff / 0.6))
    
    # Overall weighted score
    weights = {'pitch': 0.4, 'tempo': 0.3, 'dynamics': 0.3}
    overall_score = (
        weights['pitch'] * pitch_score +
        weights['tempo'] * tempo_score +
        weights['dynamics'] * dynamics_score
    )
    
    return {
        'tempo_score': round(tempo_score, 1),
        'pitch_score': round(pitch_score, 1),
        'dynamics_score': round(dynamics_score, 1),
        'overall_score': round(overall_score, 1)
    }


def extract_midi_notes(audio_path: str) -> List[Dict]:
    """
    Extract MIDI-like note events using Basic Pitch.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of note dictionaries with keys: pitch, onset_s, offset_s, confidence
    """
    if not BASIC_PITCH_AVAILABLE:
        return []
    
    try:
        # Run Basic Pitch inference
        result = predict(audio_path)
        
        # Handle different return formats
        if len(result) == 3:
            model_output, midi_data, note_events = result
        elif len(result) == 2:
            midi_data, note_events = result
        else:
            print(f"Unexpected Basic Pitch return format: {len(result)} items")
            return []
        
        # Extract note information
        notes = []
        
        # Note events can be in different formats
        if hasattr(note_events, 'notes'):
            # PrettyMIDI format
            for note in note_events.notes:
                confidence = note.velocity / 127.0
                if confidence >= MIDI_CONFIDENCE_THRESHOLD:
                    notes.append({
                        'pitch': int(note.pitch),
                        'onset_s': float(note.start),
                        'offset_s': float(note.end),
                        'confidence': float(confidence)
                    })
        elif isinstance(note_events, (list, np.ndarray)):
            # Array format
            for note_data in note_events:
                if len(note_data) >= 4:
                    onset, offset, pitch, velocity = note_data[:4]
                    confidence = velocity / 127.0
                    
                    if confidence >= MIDI_CONFIDENCE_THRESHOLD:
                        notes.append({
                            'pitch': int(pitch),
                            'onset_s': float(onset),
                            'offset_s': float(offset),
                            'confidence': float(confidence)
                        })
        
        # Sort by onset time
        notes.sort(key=lambda x: x['onset_s'])
        return notes
        
    except Exception as e:
        print(f"Warning: Basic Pitch failed for {audio_path}: {e}")
        return []


def align_midi_notes(ref_notes: List[Dict], user_notes: List[Dict]) -> Tuple[List[Tuple], Dict]:
    """
    Align MIDI note events between reference and user performances.
    
    Args:
        ref_notes: Reference note events
        user_notes: User note events
        
    Returns:
        Tuple of (matched_pairs, alignment_stats)
    """
    if not ref_notes or not user_notes:
        return [], {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Use greedy matching algorithm (simpler than Hungarian for this case)
    matched_pairs = []
    used_user_indices = set()
    
    for ref_note in ref_notes:
        best_match = None
        best_distance = float('inf')
        best_user_idx = -1
        
        for user_idx, user_note in enumerate(user_notes):
            if user_idx in used_user_indices:
                continue
                
            # Check if pitches match
            if user_note['pitch'] == ref_note['pitch']:
                # Calculate onset time difference
                onset_diff = abs(user_note['onset_s'] - ref_note['onset_s'])
                
                if onset_diff < MIDI_ONSET_TOLERANCE and onset_diff < best_distance:
                    best_match = user_note
                    best_distance = onset_diff
                    best_user_idx = user_idx
        
        if best_match is not None:
            matched_pairs.append((ref_note, best_match))
            used_user_indices.add(best_user_idx)
    
    # Calculate precision, recall, F1
    true_positives = len(matched_pairs)
    false_negatives = len(ref_notes) - true_positives
    false_positives = len(user_notes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    alignment_stats = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_ref_notes': len(ref_notes),
        'total_user_notes': len(user_notes)
    }
    
    return matched_pairs, alignment_stats


def detect_midi_issues(ref_notes: List[Dict], user_notes: List[Dict], matched_pairs: List[Tuple]) -> List[Dict]:
    """
    Detect note-level issues based on MIDI analysis.
    
    Args:
        ref_notes: Reference note events
        user_notes: User note events
        matched_pairs: Aligned note pairs
        
    Returns:
        List of MIDI-based issue dictionaries
    """
    issues = []
    
    if not ref_notes:
        return issues
    
    # Find unmatched reference notes (missed notes)
    matched_ref_notes = {id(ref_note) for ref_note, _ in matched_pairs}
    
    # Group consecutive missed notes into spans
    missed_notes = [note for note in ref_notes if id(note) not in matched_ref_notes]
    
    if missed_notes:
        # Group consecutive missed notes by time proximity
        current_group = [missed_notes[0]]
        
        for note in missed_notes[1:]:
            # If note is within 1 second of the last note in current group, add to group
            if note['onset_s'] - current_group[-1]['offset_s'] <= 1.0:
                current_group.append(note)
            else:
                # Create issue for current group and start new group
                if current_group:
                    start_time = current_group[0]['onset_s']
                    end_time = current_group[-1]['offset_s']
                    note_count = len(current_group)
                    
                    issues.append({
                        'start_s': start_time,
                        'end_s': end_time,
                        'type': 'missed_notes_midi',
                        'severity': min(2.0 + note_count * 0.5, 5.0),
                        'explanation': f'Likely missed {note_count} note(s) (MIDI estimate)'
                    })
                
                current_group = [note]
        
        # Handle last group
        if current_group:
            start_time = current_group[0]['onset_s']
            end_time = current_group[-1]['offset_s']
            note_count = len(current_group)
            
            issues.append({
                'start_s': start_time,
                'end_s': end_time,
                'type': 'missed_notes_midi',
                'severity': min(2.0 + note_count * 0.5, 5.0),
                'explanation': f'Likely missed {note_count} note(s) (MIDI estimate)'
            })
    
    # Find extra notes (user played notes not in reference)
    matched_user_notes = {id(user_note) for _, user_note in matched_pairs}
    extra_notes = [note for note in user_notes if id(note) not in matched_user_notes]
    
    if extra_notes:
        # Group consecutive extra notes
        current_group = [extra_notes[0]]
        
        for note in extra_notes[1:]:
            if note['onset_s'] - current_group[-1]['offset_s'] <= 1.0:
                current_group.append(note)
            else:
                if current_group:
                    start_time = current_group[0]['onset_s']
                    end_time = current_group[-1]['offset_s']
                    note_count = len(current_group)
                    
                    issues.append({
                        'start_s': start_time,
                        'end_s': end_time,
                        'type': 'extra_notes_midi',
                        'severity': min(1.5 + note_count * 0.3, 3.0),
                        'explanation': f'Likely extra {note_count} note(s) (MIDI estimate)'
                    })
                
                current_group = [note]
        
        # Handle last group
        if current_group:
            start_time = current_group[0]['onset_s']
            end_time = current_group[-1]['offset_s']
            note_count = len(current_group)
            
            issues.append({
                'start_s': start_time,
                'end_s': end_time,
                'type': 'extra_notes_midi',
                'severity': min(1.5 + note_count * 0.3, 3.0),
                'explanation': f'Likely extra {note_count} note(s) (MIDI estimate)'
            })
    
    return issues


def analyze_midi_performance(user_wav_path: str, ref_wav_path: str) -> Dict:
    """
    Perform MIDI-based performance analysis using piano transcription.
    
    Args:
        user_wav_path: Path to user's audio file
        ref_wav_path: Path to reference audio file
        
    Returns:
        MIDI analysis results dictionary
    """
    # Try piano transcription first (much better than Basic Pitch)
    if PIANO_TRANSCRIPTION_AVAILABLE:
        try:
            # Extract notes from both audio files using piano transcription
            ref_notes = transcribe_piano_to_notes(ref_wav_path)
            user_notes = transcribe_piano_to_notes(user_wav_path)
            
            if not ref_notes or not user_notes:
                return {
                    'available': False,
                    'reason': 'Piano transcription returned no notes'
                }
            
            # Align notes
            matched_pairs, alignment_stats = align_midi_notes(ref_notes, user_notes)
            
            # Detect issues
            midi_issues = detect_midi_issues(ref_notes, user_notes, matched_pairs)
            
            return {
                'available': True,
                'ref_notes': len(ref_notes),
                'user_notes': len(user_notes),
                'matched_notes': len(matched_pairs),
                'precision': alignment_stats['precision'],
                'recall': alignment_stats['recall'],
                'f1': alignment_stats['f1'],
                'alignment_stats': alignment_stats,
                'issues': midi_issues,
                'note_events': {
                    'reference': ref_notes,
                    'user': user_notes,
                    'matched_pairs': [(ref.copy(), u.copy()) for ref, u in matched_pairs]
                }
            }
            
        except Exception as e:
            logger.warning(f"Piano transcription failed: {e}")
    
    # Fall back to Basic Pitch if available and enabled
    basic_pitch_enabled = os.getenv('BASIC_PITCH', 'false').lower() == 'true'
    
    if not BASIC_PITCH_AVAILABLE or not basic_pitch_enabled:
        return {
            'available': False,
            'reason': 'Neither piano transcription nor Basic Pitch available'
        }
    
    try:
        # Extract notes from both audio files using Basic Pitch
        ref_notes = extract_midi_notes(ref_wav_path)
        user_notes = extract_midi_notes(user_wav_path)
        
        # Align notes
        matched_pairs, alignment_stats = align_midi_notes(ref_notes, user_notes)
        
        # Detect issues
        midi_issues = detect_midi_issues(ref_notes, user_notes, matched_pairs)
        
        return {
            'available': True,
            'ref_notes': len(ref_notes),
            'user_notes': len(user_notes),
            'matched_notes': len(matched_pairs),
            'precision': alignment_stats['precision'],
            'recall': alignment_stats['recall'],
            'f1': alignment_stats['f1'],
            'alignment_stats': alignment_stats,
            'issues': midi_issues,
            'note_events': {
                'reference': ref_notes,
                'user': user_notes,
                'matched_pairs': [(ref.copy(), user.copy()) for ref, user in matched_pairs]
            }
        }
        
    except Exception as e:
        return {
            'available': False,
            'reason': f'MIDI analysis failed: {str(e)}'
        }


def generate_tips(issues: List[Dict], overall_score: float, tempo_score: float, pitch_score: float, dynamics_score: float) -> List[str]:
    """
    Generate user-facing practice tips based on detected issues and scores.
    
    Args:
        issues: List of detected issues
        overall_score: Overall performance score (0-100)
        tempo_score: Tempo accuracy score (0-100)
        pitch_score: Pitch accuracy score (0-100)
        dynamics_score: Dynamics control score (0-100)
        
    Returns:
        List of concise practice tips (≤140 chars each)
    """
    tips = []
    
    # Analyze issue patterns
    issue_types = [issue['type'] for issue in issues]
    pitch_issues = [issue for issue in issues if issue['type'] == 'pitch_accuracy']
    tempo_issues = [issue for issue in issues if issue['type'] == 'tempo_drift']
    dynamics_issues = [issue for issue in issues if issue['type'] == 'dynamics_mismatch']
    articulation_issues = [issue for issue in issues if issue['type'] == 'messy_articulation']
    
    # Score-based tips
    if overall_score < 70:
        tips.append("Focus on accuracy first, then build speed. Practice at 70% tempo with a metronome.")
    
    if tempo_score < 75 and len(tempo_issues) > 0:
        worst_tempo_issue = max(tempo_issues, key=lambda x: x['severity'])
        start_time = worst_tempo_issue['start_s']
        tips.append(f"Practice from {start_time:.0f}s mark with metronome; focus on steady tempo.")
    
    if pitch_score < 75 and len(pitch_issues) > 0:
        # Find time range with most pitch issues
        if len(pitch_issues) > 1:
            start = min(issue['start_s'] for issue in pitch_issues)
            end = max(issue['end_s'] for issue in pitch_issues)
            tips.append(f"Review notes at {start:.0f}-{end:.0f}s; play slowly to ensure correct pitches.")
        else:
            start_time = pitch_issues[0]['start_s']
            tips.append(f"Check fingering around {start_time:.0f}s mark; practice note transitions slowly.")
    
    if dynamics_score < 75 and len(dynamics_issues) > 0:
        tips.append("Work on dynamic control; practice crescendos/diminuendos to match reference.")
    
    if len(articulation_issues) > 0:
        tips.append("Practice staccato/legato markings; ensure clean note separations and connections.")
    
    # General tips based on patterns
    if len(issues) > 3:
        tips.append("Many issues detected; break piece into sections and practice each slowly.")
    
    if not tips:  # If no specific issues, give general encouragement
        if overall_score >= 85:
            tips.append("Excellent performance! Focus on musical expression and subtle dynamics.")
        else:
            tips.append("Good foundation! Practice with metronome to tighten timing and accuracy.")
    
    # Limit to 3 most important tips and ensure they're ≤140 chars
    return [tip[:140] for tip in tips[:3]]


def save_issues_to_csv(issues: List[Dict], csv_path: str) -> None:
    """
    Save detected issues to CSV file.
    
    Args:
        issues: List of issue dictionaries
        csv_path: Output CSV file path
    """
    fieldnames = ['start_s', 'end_s', 'type', 'severity', 'explanation']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for issue in issues:
            writer.writerow({
                'start_s': f"{issue['start_s']:.2f}",
                'end_s': f"{issue['end_s']:.2f}",
                'type': issue['type'],
                'severity': f"{issue['severity']:.2f}",
                'explanation': issue['explanation']
            })


def analyze_audio_pair(user_wav_path: str, ref_wav_path: str) -> Dict:
    """
    Comprehensive analysis of user vs reference audio performance.
    
    Args:
        user_wav_path: Path to user's performance WAV file
        ref_wav_path: Path to reference performance WAV file
        
    Returns:
        Analysis results dictionary
    """
    # Import auto-trimming functions (temporarily disabled)
    # from app.utils import estimate_music_bounds, subsequence_align_bounds
    
    # 1. Load audio files
    y_user, duration_user = load_audio(user_wav_path)
    y_ref, duration_ref = load_audio(ref_wav_path)
    
    # 2. Auto-trim user audio to remove noise and applause (TEMPORARILY DISABLED)
    # logger.info("Estimating music bounds for user audio...")
    # t_start_coarse, t_end_coarse = estimate_music_bounds(y_user, TARGET_SR)
    
    # Try subsequence alignment if reference is available
    # t_start_final = t_start_coarse
    # t_end_final = t_end_coarse
    # trim_confidence = "low"
    
    # try:
    #     logger.info("Attempting subsequence alignment with reference...")
    #     t_start_aligned, t_end_aligned = subsequence_align_bounds(y_user, TARGET_SR, y_ref)
        
    #     # Validate aligned bounds
    #     aligned_duration = t_end_aligned - t_start_aligned
    #     if aligned_duration >= 10.0 and aligned_duration <= duration_user:
    #         t_start_final = t_start_aligned
    #         t_end_final = t_end_aligned
    #         trim_confidence = "high"
    #         logger.info(f"Subsequence alignment successful: {t_start_final:.2f}s to {t_end_final:.2f}s")
    #     else:
    #         logger.warning(f"Aligned bounds invalid (duration: {aligned_duration:.2f}s), using coarse bounds")
    # except Exception as e:
    #     logger.warning(f"Subsequence alignment failed: {e}, using coarse bounds")
    
    # Validate final bounds
    # final_duration = t_end_final - t_start_final
    # if final_duration < 10.0 or final_duration > duration_user:
    #     logger.warning("Final bounds invalid, using full audio")
    #     t_start_final = 0.0
    #     t_end_final = duration_user
    #     trim_confidence = "none"
    
    # Trim user audio
    # start_sample = int(t_start_final * TARGET_SR)
    # end_sample = int(t_end_final * TARGET_SR)
    # start_sample = max(0, min(start_sample, len(y_user) - 1))
    # end_sample = max(start_sample + 1, min(end_sample, len(y_user)))
    
    # y_user_trimmed = y_user[start_sample:end_sample]
    # duration_user_trimmed = len(y_user_trimmed) / TARGET_SR
    
    # logger.info(f"User audio trimmed: {t_start_final:.2f}s to {t_end_final:.2f}s (duration: {duration_user_trimmed:.2f}s, confidence: {trim_confidence})")
    
    # Use full audio for now (temporarily disable trimming)
    y_user_trimmed = y_user
    duration_user_trimmed = duration_user
    t_start_final = 0.0
    t_end_final = duration_user
    trim_confidence = "disabled"
    
    # 3. Extract features from full user audio and reference
    features_user = extract_features(y_user_trimmed, TARGET_SR, user_wav_path)
    features_ref = extract_features(y_ref, TARGET_SR, ref_wav_path)
    
    # 3. Align using DTW on chroma features
    try:
        D, wp = align_with_dtw(features_ref['chroma'], features_user['chroma'])
    except ValueError as e:
        # If DTW fails, create a simple linear alignment
        logger.warning(f"DTW alignment failed: {e}. Using linear alignment.")
        min_frames = min(features_ref['chroma'].shape[1], features_user['chroma'].shape[1])
        wp = np.column_stack([np.arange(min_frames), np.arange(min_frames)])
        D = np.zeros((min_frames, min_frames))  # Dummy distance matrix
    
    # 4. Resample to common timebase
    aligned_features = resample_to_common_timebase(features_ref, features_user, wp)
    
    # 5. Detect issues
    issues = detect_issues(aligned_features, features_ref, features_user)
    
    # 6. Compute scores
    scores = compute_scores(aligned_features, issues)
    
    # 7. MIDI analysis (optional)
    midi_results = analyze_midi_performance(user_wav_path, ref_wav_path)
    
    # Add MIDI issues to main issues list if available
    if midi_results.get('available', False) and midi_results.get('issues'):
        issues.extend(midi_results['issues'])
    
    # 8. Prepare tempo curves for output
    beat_times_ref, tempo_curve_ref = compute_tempo_curve(features_ref['beats'])
    beat_times_user, tempo_curve_user = compute_tempo_curve(features_user['beats'])
    
    # Interpolate tempo curves to common timebase
    tempo_ref_interp = np.interp(aligned_features['time_s'], beat_times_ref, tempo_curve_ref)
    tempo_user_interp = np.interp(aligned_features['time_s'], beat_times_user, tempo_curve_user)
    
    # 9. Generate practice tips
    tips = generate_tips(issues, scores['overall_score'], scores['tempo_score'], scores['pitch_score'], scores['dynamics_score'])
    
    # Add trimming note if confidence is low
    if trim_confidence == "low":
        tips.insert(0, "Auto-trim low confidence; analyzed full audio.")
    elif trim_confidence == "none":
        tips.insert(0, "Could not auto-trim; analyzed full audio.")
    elif trim_confidence == "disabled":
        tips.insert(0, "Auto-trimming temporarily disabled for stability.")
    
    # 10. Save issues to CSV
    import tempfile
    csv_path = tempfile.mktemp(suffix='_analysis_issues.csv', prefix='piano_')
    save_issues_to_csv(issues, csv_path)
    
    # 11. Compile results
    result = {
        'summary': {
            'user_duration': duration_user,
            'user_duration_trimmed': duration_user_trimmed,
            'ref_duration': duration_ref,
            'trim_start': t_start_final,
            'trim_end': t_end_final,
            'trim_confidence': trim_confidence,
            'alignment_quality': float(1.0 - np.mean(D[wp[:, 0], wp[:, 1]])),
            'total_issues': len(issues),
            'csv_path': csv_path,
            **scores
        },
        'per_frame': {
            'time_s': aligned_features['time_s'].tolist(),
            'chroma_distance': aligned_features['chroma_distance'].tolist(),
            'tempo_user': tempo_user_interp.tolist(),
            'tempo_ref': tempo_ref_interp.tolist(),
            'rms_user': aligned_features['rms_user'].tolist(),
            'rms_ref': aligned_features['rms_ref'].tolist()
        },
        'issues': issues,
        'tips': tips,
        'midi_estimates': midi_results
    }
    
    return result


# Smoke test function
def smoke_test():
    """
    Internal smoke test using sample files if present.
    """
    import tempfile
    
    print("Running smoke test...")
    
    # Create synthetic test audio
    sr = TARGET_SR
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Reference: simple melody
    freqs_ref = [440, 493.88, 523.25, 587.33, 659.25]  # A4, B4, C5, D5, E5
    y_ref = np.zeros_like(t)
    note_duration = duration / len(freqs_ref)
    
    for i, freq in enumerate(freqs_ref):
        start_idx = int(i * note_duration * sr)
        end_idx = int((i + 1) * note_duration * sr)
        if end_idx <= len(t):
            y_ref[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    
    # User: same melody but with slight variations
    y_user = y_ref.copy()
    
    # Add slight pitch variation (simulate minor inaccuracy)
    for i in range(len(y_user)):
        y_user[i] *= (1 + 0.05 * np.sin(2 * np.pi * 2 * t[i]))  # 5% vibrato
    
    # Add tempo variation (slight time stretch in middle)
    mid_start = len(y_user) // 3
    mid_end = 2 * len(y_user) // 3
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as user_file:
        
        import soundfile as sf
        sf.write(ref_file.name, y_ref, sr)
        sf.write(user_file.name, y_user, sr)
        
        try:
            # Run analysis
            result = analyze_audio_pair(user_file.name, ref_file.name)
            
            print(f"Analysis completed successfully!")
            print(f"Overall score: {result['summary']['overall_score']}")
            print(f"Issues detected: {result['summary']['total_issues']}")
            print(f"Alignment quality: {result['summary']['alignment_quality']:.3f}")
            
            # Test CSV export
            csv_path = tempfile.mktemp(suffix='.csv')
            save_issues_to_csv(result['issues'], csv_path)
            print(f"CSV exported to: {csv_path}")
            
            return True
            
        except Exception as e:
            print(f"Smoke test failed: {e}")
            return False
        finally:
            # Cleanup
            os.unlink(ref_file.name)
            os.unlink(user_file.name)


if __name__ == "__main__":
    smoke_test()
