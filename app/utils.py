#!/usr/bin/env python3
"""
Audio processing utilities for YouTube downloads and audio standardization.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple
import librosa
import soundfile as sf
import yt_dlp
from pydub import AudioSegment
from fastapi import HTTPException
import numpy as np
from scipy.signal import medfilt


# Constants
MAX_DURATION_SECONDS = 600  # 10 minutes
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
TARGET_SAMPLE_RATE = 22050


def safe_tmp_path(suffix: str = ".wav") -> str:
    """
    Generate a safe temporary file path with unique identifier.
    
    Args:
        suffix: File extension suffix (default: ".wav")
        
    Returns:
        Absolute path to a unique temporary file
    """
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    filename = f"audio_{unique_id}{suffix}"
    return os.path.join(temp_dir, filename)


def download_youtube_audio(url: str, output_path: Optional[str] = None) -> str:
    """
    Download audio from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube URL to download
        output_path: Optional output path. If None, generates a temporary path.
        
    Returns:
        Path to the downloaded audio file
        
    Raises:
        HTTPException: If download fails or audio exceeds duration limits
    """
    if output_path is None:
        output_path = safe_tmp_path(".%(ext)s")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'noplaylist': True,
        'extract_flat': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to check duration
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            
            if duration and duration > MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio duration {duration}s exceeds maximum {MAX_DURATION_SECONDS}s"
                )
            
            # Download the audio
            ydl.download([url])
            
            # Find the actual downloaded file (yt-dlp may change the extension)
            base_path = output_path.replace('.%(ext)s', '')
            for ext in ['.webm', '.m4a', '.mp3', '.wav', '.ogg']:
                potential_path = base_path + ext
                if os.path.exists(potential_path):
                    return potential_path
                    
            # If no file found, use the original path
            return output_path
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except yt_dlp.DownloadError as e:
        raise HTTPException(status_code=400, detail=f"YouTube download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during download: {str(e)}")


def verify_file_size(file_path: str) -> None:
    """
    Verify that file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        
    Raises:
        HTTPException: If file size exceeds maximum allowed size
    """
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File size {file_size} bytes exceeds maximum {MAX_FILE_SIZE_BYTES} bytes"
            )


def convert_to_wav_with_pydub(input_path: str, output_path: str) -> None:
    """
    Convert audio file to WAV format using pydub.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        
    Raises:
        HTTPException: If conversion fails
    """
    try:
        # Load audio with pydub (supports many formats)
        audio = AudioSegment.from_file(input_path)
        
        # Export as WAV
        audio.export(output_path, format="wav")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio conversion to WAV failed: {str(e)}"
        )


def standardize_audio_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = TARGET_SAMPLE_RATE
) -> Tuple[str, float]:
    """
    Standardize audio file to WAV format with specified sample rate and mono channel.
    
    Args:
        input_path: Path to input audio file
        output_path: Optional output path. If None, generates a temporary path.
        sample_rate: Target sample rate (default: 22050 Hz)
        
    Returns:
        Tuple of (output_path, duration_seconds)
        
    Raises:
        HTTPException: If processing fails or duration exceeds limits
    """
    if output_path is None:
        output_path = safe_tmp_path(".wav")
    
    try:
        # Verify input file size first
        verify_file_size(input_path)
        
        # Convert to WAV if not already WAV
        temp_wav_path = None
        if not input_path.lower().endswith('.wav'):
            temp_wav_path = safe_tmp_path(".wav")
            convert_to_wav_with_pydub(input_path, temp_wav_path)
            wav_input_path = temp_wav_path
        else:
            wav_input_path = input_path
        
        # Load audio with librosa (this handles sample rate conversion)
        audio_data, sr = librosa.load(wav_input_path, sr=sample_rate, mono=True)
        
        # Verify duration using librosa
        duration = librosa.get_duration(y=audio_data, sr=sr)
        if duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=f"Audio duration {duration:.2f}s exceeds maximum {MAX_DURATION_SECONDS}s"
            )
        
        # Write standardized audio using soundfile
        sf.write(output_path, audio_data, sample_rate)
        
        # Clean up temporary WAV file if we created one
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        
        return output_path, duration
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Clean up temporary files on error
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Audio standardization failed: {str(e)}"
        )


def process_youtube_url(url: str, output_path: Optional[str] = None) -> Tuple[str, float]:
    """
    Complete pipeline: download YouTube audio and standardize it.
    
    Args:
        url: YouTube URL to process
        output_path: Optional output path for final WAV file
        
    Returns:
        Tuple of (final_wav_path, duration_seconds)
        
    Raises:
        HTTPException: If any step in the pipeline fails
    """
    downloaded_path = None
    try:
        # Download from YouTube
        downloaded_path = download_youtube_audio(url)
        
        # Standardize the audio
        final_path, duration = standardize_audio_to_wav(downloaded_path, output_path)
        
        return final_path, duration
        
    finally:
        # Clean up downloaded file
        if downloaded_path and os.path.exists(downloaded_path):
            try:
                os.unlink(downloaded_path)
            except:
                pass


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely remove a temporary file.
    
    Args:
        file_path: Path to the file to remove
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        # Ignore cleanup errors
        pass


def estimate_music_bounds(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Estimate the start and end times of actual music content, excluding noise and applause.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    import librosa
    
    # Parameters
    rms_threshold_db = -45  # dBFS threshold for music detection
    min_region_length = 2.0  # Minimum music region length in seconds
    gap_bridge_length = 0.6  # Maximum gap to bridge in seconds
    harmonic_ratio_threshold = 0.35  # Minimum harmonic energy ratio
    flatness_threshold = 0.5  # Maximum spectral flatness
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    # Convert to dB and smooth
    rms_db = 20 * np.log10(rms + 1e-8)
    rms_db_smooth = medfilt(rms_db, kernel_size=5)
    
    # Create initial music mask based on RMS threshold
    music_mask = rms_db_smooth > rms_threshold_db
    
    # Apply HPSS to separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Compute harmonic energy ratio
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=512)[0]
    rms_percussive = librosa.feature.rms(y=y_percussive, hop_length=512)[0]
    harmonic_energy_ratio = rms_harmonic / (rms_harmonic + rms_percussive + 1e-8)
    
    # Compute spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
    
    # Refine music mask using harmonic content and flatness
    harmonic_mask = harmonic_energy_ratio > harmonic_ratio_threshold
    flatness_mask = spectral_flatness < flatness_threshold
    music_mask = music_mask & harmonic_mask & flatness_mask
    
    # Find continuous regions
    region_starts = []
    region_ends = []
    in_region = False
    region_start = 0
    
    for i, is_music in enumerate(music_mask):
        if is_music and not in_region:
            region_start = frame_times[i]
            in_region = True
        elif not is_music and in_region:
            region_end = frame_times[i]
            if region_end - region_start >= min_region_length:
                region_starts.append(region_start)
                region_ends.append(region_end)
            in_region = False
    
    # Handle case where we end in a music region
    if in_region:
        region_end = frame_times[-1]
        if region_end - region_start >= min_region_length:
            region_starts.append(region_start)
            region_ends.append(region_end)
    
    # Bridge small gaps between regions
    if len(region_starts) > 1:
        bridged_starts = [region_starts[0]]
        bridged_ends = []
        
        for i in range(len(region_starts) - 1):
            gap = region_starts[i + 1] - region_ends[i]
            if gap <= gap_bridge_length:
                # Bridge the gap
                bridged_ends.append(region_ends[i + 1])
            else:
                # Don't bridge, end current region
                bridged_ends.append(region_ends[i])
                bridged_starts.append(region_starts[i + 1])
        
        # Handle last region
        if len(bridged_ends) < len(bridged_starts):
            bridged_ends.append(region_ends[-1])
        
        region_starts = bridged_starts
        region_ends = bridged_ends
    
    # Return bounds
    if region_starts and region_ends:
        t_start = region_starts[0]
        t_end = region_ends[-1]
    else:
        # Fallback: return full duration
        t_start = 0.0
        t_end = librosa.get_duration(y=y, sr=sr)
    
    return t_start, t_end


def subsequence_align_bounds(user_y: np.ndarray, sr: int, ref_y: np.ndarray) -> Tuple[float, float]:
    """
    Find the best matching subsequence of user audio to the full reference using DTW.
    
    Args:
        user_y: User audio time series
        sr: Sample rate
        ref_y: Reference audio time series
        
    Returns:
        Tuple of (start_time, end_time) for user audio
    """
    import librosa
    
    # Parameters
    hop_length = 512
    sakoe_chiba_band = 0.12  # 12% band for DTW
    
    # Compute beat-synchronous chroma for both signals
    def get_beat_chroma(y, sr):
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Beat tracking
        beats, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        # Ensure we have beats
        if len(beats) == 0:
            duration = librosa.get_duration(y=y, sr=sr)
            beats = np.arange(0, duration, 0.5)  # Beat every 0.5 seconds
        
        # Convert beats to frame indices
        beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
        
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        
        # Extract chroma at beat frames
        beat_chroma = chroma[:, beat_frames]
        
        return beat_chroma, beats
    
    # Get beat-synchronous chroma for both signals
    ref_chroma, ref_beats = get_beat_chroma(ref_y, sr)
    user_chroma, user_beats = get_beat_chroma(user_y, sr)
    
    # Handle NaN values
    ref_chroma = np.nan_to_num(ref_chroma, nan=0.0)
    user_chroma = np.nan_to_num(user_chroma, nan=0.0)
    
    # Normalize chroma features
    ref_chroma_norm = ref_chroma / (np.linalg.norm(ref_chroma, axis=0, keepdims=True) + 1e-8)
    user_chroma_norm = user_chroma / (np.linalg.norm(user_chroma, axis=0, keepdims=True) + 1e-8)
    
    try:
        # Run subsequence DTW
        D, wp = librosa.sequence.dtw(
            X=ref_chroma_norm, 
            Y=user_chroma_norm, 
            metric='cosine',
            subseq=True,
            step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]),
            weights_add=np.array([0, 0, 0]),
            weights_mul=np.array([1, 1, 1])
        )
        
        # Find the best path through the cost matrix
        # For subsequence DTW, we want to find the best alignment of user to reference
        best_path = librosa.sequence.dtw_backtrack(D, wp)
        
        # Extract user indices from the path
        user_indices = best_path[:, 1]
        
        # Convert to beat indices and then to times
        user_start_beat_idx = user_indices[0]
        user_end_beat_idx = user_indices[-1]
        
        # Ensure indices are within bounds
        user_start_beat_idx = max(0, min(user_start_beat_idx, len(user_beats) - 1))
        user_end_beat_idx = max(0, min(user_end_beat_idx, len(user_beats) - 1))
        
        t_start = user_beats[user_start_beat_idx]
        t_end = user_beats[user_end_beat_idx]
        
        return t_start, t_end
        
    except Exception as e:
        # If DTW fails, fall back to full duration
        duration = librosa.get_duration(y=user_y, sr=sr)
        return 0.0, duration


def trim_to_bounds(in_wav_path: str, out_wav_path: str, t0: float, t1: float) -> None:
    """
    Trim audio file to specified time bounds with fade in/out.
    
    Args:
        in_wav_path: Input WAV file path
        out_wav_path: Output WAV file path
        t0: Start time in seconds
        t1: End time in seconds
    """
    import librosa
    import soundfile as sf
    
    # Load audio
    y, sr = librosa.load(in_wav_path, sr=22050, mono=True)
    
    # Convert times to sample indices
    start_sample = int(t0 * sr)
    end_sample = int(t1 * sr)
    
    # Ensure bounds are valid
    start_sample = max(0, min(start_sample, len(y) - 1))
    end_sample = max(start_sample + 1, min(end_sample, len(y)))
    
    # Extract the segment
    y_trimmed = y[start_sample:end_sample]
    
    # Apply fade in/out (10ms)
    fade_samples = int(0.01 * sr)  # 10ms fade
    fade_samples = min(fade_samples, len(y_trimmed) // 2)  # Don't fade more than half the audio
    
    if fade_samples > 0:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        y_trimmed[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        y_trimmed[-fade_samples:] *= fade_out
    
    # Write trimmed audio
    sf.write(out_wav_path, y_trimmed, sr)
