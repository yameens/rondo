"""Utility functions for audio analysis and processing."""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that an audio file exists and is readable.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        # Try to load a small portion to validate format
        y, sr = librosa.load(audio_path, sr=None, duration=0.1)
        
        if len(y) == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Invalid audio file {audio_path}: {e}")
        return False


def validate_musicxml_file(musicxml_path: str) -> bool:
    """
    Validate that a MusicXML file exists and is readable.
    
    Args:
        musicxml_path: Path to MusicXML file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(musicxml_path):
            logger.error(f"MusicXML file not found: {musicxml_path}")
            return False
        
        # Check file extension
        path = Path(musicxml_path)
        if path.suffix.lower() not in ['.xml', '.musicxml', '.mxl']:
            logger.warning(f"Unexpected MusicXML file extension: {path.suffix}")
        
        # Try to read the file
        with open(musicxml_path, 'r', encoding='utf-8') as f:
            content = f.read(100)  # Read first 100 chars
            if 'xml' not in content.lower():
                logger.warning(f"File may not be valid XML: {musicxml_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Invalid MusicXML file {musicxml_path}: {e}")
        return False


def normalize_beat_positions(beat_times: List[float]) -> List[float]:
    """
    Normalize beat positions to [0, 1] range.
    
    Args:
        beat_times: Absolute beat times in seconds
        
    Returns:
        Normalized beat positions
    """
    if not beat_times:
        return []
    
    min_time = min(beat_times)
    max_time = max(beat_times)
    
    if max_time <= min_time:
        # All beats at same time: return evenly spaced
        return [i / max(len(beat_times) - 1, 1) for i in range(len(beat_times))]
    
    return [(t - min_time) / (max_time - min_time) for t in beat_times]


def interpolate_to_beat_grid(
    values: np.ndarray, 
    value_times: np.ndarray, 
    beat_times: List[float]
) -> List[float]:
    """
    Interpolate time-series values to beat grid positions.
    
    Args:
        values: Array of feature values
        value_times: Time points for each value
        beat_times: Target beat time positions
        
    Returns:
        Interpolated values at beat positions
    """
    try:
        if len(values) == 0 or len(value_times) == 0:
            return [0.0] * len(beat_times)
        
        # Ensure arrays are sorted by time
        sort_idx = np.argsort(value_times)
        value_times_sorted = value_times[sort_idx]
        values_sorted = values[sort_idx]
        
        # Interpolate to beat positions
        interpolated = np.interp(beat_times, value_times_sorted, values_sorted)
        
        return interpolated.tolist()
        
    except Exception as e:
        logger.error(f"Failed to interpolate to beat grid: {e}")
        return [0.0] * len(beat_times)


def aggregate_values_per_beat(
    values: np.ndarray,
    value_times: np.ndarray,
    beat_intervals: List[Tuple[float, float]],
    aggregation: str = 'mean'
) -> List[float]:
    """
    Aggregate time-series values within beat intervals.
    
    Args:
        values: Array of feature values
        value_times: Time points for each value
        beat_intervals: List of (start_time, end_time) tuples for each beat
        aggregation: Aggregation method ('mean', 'median', 'max', 'rms')
        
    Returns:
        Aggregated values per beat interval
    """
    try:
        aggregated = []
        
        for start_time, end_time in beat_intervals:
            # Find values within this interval
            mask = (value_times >= start_time) & (value_times < end_time)
            interval_values = values[mask]
            
            if len(interval_values) > 0:
                if aggregation == 'mean':
                    agg_value = np.mean(interval_values)
                elif aggregation == 'median':
                    agg_value = np.median(interval_values)
                elif aggregation == 'max':
                    agg_value = np.max(interval_values)
                elif aggregation == 'rms':
                    agg_value = np.sqrt(np.mean(interval_values ** 2))
                else:
                    agg_value = np.mean(interval_values)  # Default to mean
            else:
                agg_value = 0.0
            
            aggregated.append(float(agg_value))
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Failed to aggregate values per beat: {e}")
        return [0.0] * len(beat_intervals)


def smooth_curve(values: List[float], window_size: int = 3) -> List[float]:
    """
    Apply smoothing to a feature curve.
    
    Args:
        values: Input values
        window_size: Size of smoothing window (odd number)
        
    Returns:
        Smoothed values
    """
    try:
        if len(values) < window_size:
            return values
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(values)):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(values), i + half_window + 1)
            
            # Compute mean within window
            window_values = values[start_idx:end_idx]
            smoothed.append(sum(window_values) / len(window_values))
        
        return smoothed
        
    except Exception as e:
        logger.error(f"Failed to smooth curve: {e}")
        return values


def detect_outliers(values: List[float], threshold: float = 2.0) -> List[bool]:
    """
    Detect outliers in a feature curve using z-score.
    
    Args:
        values: Input values
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean mask indicating outliers
    """
    try:
        if len(values) < 3:
            return [False] * len(values)
        
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        if std_val == 0:
            return [False] * len(values)
        
        z_scores = np.abs((values_array - mean_val) / std_val)
        outliers = z_scores > threshold
        
        return outliers.tolist()
        
    except Exception as e:
        logger.error(f"Failed to detect outliers: {e}")
        return [False] * len(values)


def create_beat_intervals(beat_times: List[float]) -> List[Tuple[float, float]]:
    """
    Create time intervals from beat positions.
    
    Args:
        beat_times: Beat time positions
        
    Returns:
        List of (start_time, end_time) intervals
    """
    if len(beat_times) < 2:
        return [(0.0, 1.0)]
    
    intervals = []
    for i in range(len(beat_times) - 1):
        intervals.append((beat_times[i], beat_times[i + 1]))
    
    # Add final interval (from last beat to estimated end)
    last_duration = beat_times[-1] - beat_times[-2] if len(beat_times) > 1 else 1.0
    intervals.append((beat_times[-1], beat_times[-1] + last_duration))
    
    return intervals


def validate_feature_curve_lengths(curves: Dict[str, List[float]]) -> bool:
    """
    Validate that all feature curves have the same length.
    
    Args:
        curves: Dictionary of feature name -> values
        
    Returns:
        True if all curves have same length, False otherwise
    """
    if not curves:
        return True
    
    lengths = [len(values) for values in curves.values()]
    
    if len(set(lengths)) > 1:
        logger.error(f"Feature curve length mismatch: {dict(zip(curves.keys(), lengths))}")
        return False
    
    return True


def pad_or_truncate_curve(values: List[float], target_length: int, pad_value: float = 0.0) -> List[float]:
    """
    Pad or truncate a curve to target length.
    
    Args:
        values: Input values
        target_length: Desired length
        pad_value: Value to use for padding
        
    Returns:
        Adjusted values
    """
    if len(values) == target_length:
        return values
    elif len(values) < target_length:
        # Pad with last value or pad_value
        pad_val = values[-1] if values else pad_value
        return values + [pad_val] * (target_length - len(values))
    else:
        # Truncate
        return values[:target_length]
