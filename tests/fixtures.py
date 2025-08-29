"""
Test fixtures for generating synthetic audio data for analysis tests.
"""
import numpy as np
import soundfile as sf
import tempfile
import os
from typing import List, Tuple, Optional
from scipy import signal


TARGET_SR = 22050
TEMP_DIR = tempfile.gettempdir()


def generate_sine_wave(frequency: float, duration: float, amplitude: float = 0.5, 
                      sample_rate: int = TARGET_SR) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_chord(frequencies: List[float], duration: float, amplitude: float = 0.5,
                  sample_rate: int = TARGET_SR) -> np.ndarray:
    """Generate a chord by combining multiple sine waves."""
    chord = np.zeros(int(sample_rate * duration))
    for freq in frequencies:
        chord += generate_sine_wave(freq, duration, amplitude / len(frequencies), sample_rate)
    return chord


def generate_c_major_scale(duration_per_note: float = 0.5, amplitude: float = 0.5,
                          sample_rate: int = TARGET_SR) -> np.ndarray:
    """Generate a C major scale (C4 to C5)."""
    # C major scale frequencies (C4 to C5)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    scale = np.array([])
    for freq in frequencies:
        note = generate_sine_wave(freq, duration_per_note, amplitude, sample_rate)
        # Add small gap between notes
        gap = np.zeros(int(sample_rate * 0.1))
        scale = np.concatenate([scale, note, gap])
    
    return scale


def generate_piano_like_tone(frequency: float, duration: float, amplitude: float = 0.5,
                           sample_rate: int = TARGET_SR) -> np.ndarray:
    """Generate a more realistic piano-like tone with harmonics."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Fundamental frequency
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics with decreasing amplitude
    harmonics = [2, 3, 4, 5]
    harmonic_amps = [0.3, 0.2, 0.1, 0.05]
    
    for harmonic, harm_amp in zip(harmonics, harmonic_amps):
        tone += amplitude * harm_amp * np.sin(2 * np.pi * frequency * harmonic * t)
    
    # Apply envelope (attack, decay, sustain, release)
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)  # 0.1s attack
    release_samples = int(0.2 * sample_rate)  # 0.2s release
    
    if len(envelope) > attack_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if len(envelope) > release_samples:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    return tone * envelope


def time_stretch_audio(audio: np.ndarray, stretch_factor: float) -> np.ndarray:
    """
    Time-stretch audio by the given factor.
    stretch_factor > 1.0 makes audio slower/longer
    stretch_factor < 1.0 makes audio faster/shorter
    """
    # Simple linear interpolation time stretching
    original_length = len(audio)
    new_length = int(original_length * stretch_factor)
    
    # Create new time indices
    old_indices = np.linspace(0, original_length - 1, new_length)
    
    # Interpolate
    stretched = np.interp(old_indices, np.arange(original_length), audio)
    
    # Pad or truncate to roughly match original length for easier comparison
    if len(stretched) > original_length:
        return stretched[:original_length]
    else:
        padded = np.zeros(original_length)
        padded[:len(stretched)] = stretched
        return padded


def pitch_shift_audio(audio: np.ndarray, semitones: float, 
                     sample_rate: int = TARGET_SR) -> np.ndarray:
    """
    Pitch shift audio by the given number of semitones.
    Positive values shift up, negative values shift down.
    """
    # Calculate the pitch shift factor
    shift_factor = 2 ** (semitones / 12.0)
    
    # Simple approach: resample and then time-stretch back
    # This changes both pitch and tempo, then we correct tempo
    
    # First, resample to change both pitch and tempo
    new_length = int(len(audio) / shift_factor)
    old_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(old_indices, np.arange(len(audio)), audio)
    
    # Then time-stretch back to original tempo
    time_stretched = time_stretch_audio(resampled, shift_factor)
    
    # Trim or pad to match original length approximately
    if len(time_stretched) > len(audio):
        return time_stretched[:len(audio)]
    else:
        padded = np.zeros(len(audio))
        padded[:len(time_stretched)] = time_stretched
        return padded


def amplify_region(audio: np.ndarray, start_time: float, end_time: float,
                  amplification_factor: float, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Amplify a specific time region of the audio."""
    audio_copy = audio.copy()
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    audio_copy[start_sample:end_sample] *= amplification_factor
    
    return audio_copy


def save_test_audio(audio: np.ndarray, filename: str, 
                   sample_rate: int = TARGET_SR) -> str:
    """Save audio data to a temporary WAV file and return the path."""
    filepath = os.path.join(TEMP_DIR, filename)
    
    # Ensure audio is in valid range [-1, 1]
    audio_normalized = np.clip(audio, -1.0, 1.0)
    
    sf.write(filepath, audio_normalized, sample_rate)
    return filepath


def cleanup_test_file(filepath: str) -> None:
    """Remove a test file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass  # Ignore cleanup errors


class AudioTestFixture:
    """Context manager for creating and cleaning up test audio files."""
    
    def __init__(self, audio: np.ndarray, filename: str, 
                 sample_rate: int = TARGET_SR):
        self.audio = audio
        self.filename = filename
        self.sample_rate = sample_rate
        self.filepath = None
    
    def __enter__(self) -> str:
        self.filepath = save_test_audio(self.audio, self.filename, self.sample_rate)
        return self.filepath
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filepath:
            cleanup_test_file(self.filepath)


def create_test_reference_audio(duration: float = 4.0) -> np.ndarray:
    """
    Create a reference audio for testing - a short musical sequence.
    Duration should be kept short for fast tests.
    """
    # Create a simple melody with chord progression
    # Each chord lasts 1 second
    chord_duration = duration / 4
    
    # C major, F major, G major, C major progression
    chords = [
        [261.63, 329.63, 392.00],  # C major (C, E, G)
        [174.61, 220.00, 261.63],  # F major (F, A, C) - lower octave
        [196.00, 246.94, 293.66],  # G major (G, B, D)
        [261.63, 329.63, 392.00],  # C major (C, E, G)
    ]
    
    reference = np.array([])
    for chord_freqs in chords:
        chord = generate_chord(chord_freqs, chord_duration, amplitude=0.3)
        reference = np.concatenate([reference, chord])
    
    return reference


def create_modified_test_audio(base_audio: np.ndarray, modification_type: str,
                              **kwargs) -> np.ndarray:
    """
    Create modified version of base audio for testing specific issues.
    
    Args:
        base_audio: The original audio to modify
        modification_type: Type of modification ('tempo', 'pitch', 'dynamics')
        **kwargs: Additional parameters for the modification
    """
    if modification_type == 'tempo':
        stretch_factor = kwargs.get('stretch_factor', 1.05)  # 5% slower
        return time_stretch_audio(base_audio, stretch_factor)
    
    elif modification_type == 'pitch':
        semitones = kwargs.get('semitones', 2)  # +2 semitones
        start_time = kwargs.get('start_time', 1.0)  # Apply to middle section
        end_time = kwargs.get('end_time', 3.0)
        
        # Only pitch shift a portion of the audio
        modified = base_audio.copy()
        start_sample = int(start_time * TARGET_SR)
        end_sample = int(end_time * TARGET_SR)
        
        if start_sample < len(modified) and end_sample <= len(modified):
            segment = modified[start_sample:end_sample]
            shifted_segment = pitch_shift_audio(segment, semitones)
            modified[start_sample:end_sample] = shifted_segment
        
        return modified
    
    elif modification_type == 'dynamics':
        amplification = kwargs.get('amplification', 3.0)  # 3x louder
        start_time = kwargs.get('start_time', 1.5)
        end_time = kwargs.get('end_time', 2.5)
        
        return amplify_region(base_audio, start_time, end_time, amplification)
    
    else:
        raise ValueError(f"Unknown modification type: {modification_type}")
