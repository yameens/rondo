"""
Professional onset and frame detection for piano performance analysis.
Now uses ByteDance Piano Transcription as primary method with librosa fallback.
"""

import numpy as np
import librosa
import scipy.signal
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import ByteDance Piano Transcription (primary method)
try:
    from .piano_transcription import (
        analyze_piano_performance,
        transcribe_piano_to_notes,
        extract_onset_times,
        extract_beat_times,
        PianoTranscriptionError
    )
    BYTEDANCE_AVAILABLE = True
    logger.info("ByteDance Piano Transcription available - using as primary method")
except ImportError as e:
    BYTEDANCE_AVAILABLE = False
    logger.warning(f"ByteDance Piano Transcription not available: {e}")

# Basic Pitch as fallback only
try:
    import basic_pitch
    from basic_pitch.inference import predict
    BASIC_PITCH_AVAILABLE = True
    logger.info("Basic Pitch available as fallback")
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    logger.warning("Basic Pitch not available")

class OnsetFrameDetector:
    """
    High-quality onset and frame detection for piano audio.
    Uses multiple detection methods and combines them for robust results.
    """
    
    def __init__(self, 
                 sr: int = 22050,
                 hop_length: int = 512,
                 frame_length: int = 2048):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.frame_time = hop_length / sr
        
    def detect_onsets_multi_method(self, 
                                   audio: np.ndarray,
                                   combine_method: str = 'weighted') -> np.ndarray:
        """
        Detect onsets using multiple methods and combine results.
        
        Args:
            audio: Audio signal
            combine_method: How to combine methods ('weighted', 'union', 'intersection')
            
        Returns:
            Array of onset times in seconds
        """
        methods = {
            'spectral_flux': self._onset_spectral_flux,
            'complex_domain': self._onset_complex_domain,
            'energy_based': self._onset_energy_based,
            'phase_deviation': self._onset_phase_deviation,
            'high_frequency': self._onset_high_frequency
        }
        
        onset_frames = {}
        onset_strengths = {}
        
        for method_name, method_func in methods.items():
            try:
                frames, strengths = method_func(audio)
                onset_frames[method_name] = frames
                onset_strengths[method_name] = strengths
                logger.debug(f"{method_name}: detected {len(frames)} onsets")
            except Exception as e:
                logger.warning(f"Onset detection method {method_name} failed: {e}")
                continue
        
        if not onset_frames:
            logger.error("All onset detection methods failed")
            return np.array([])
        
        # Combine onset detections
        if combine_method == 'weighted':
            combined_onsets = self._combine_onsets_weighted(onset_frames, onset_strengths)
        elif combine_method == 'union':
            combined_onsets = self._combine_onsets_union(onset_frames)
        elif combine_method == 'intersection':
            combined_onsets = self._combine_onsets_intersection(onset_frames)
        else:
            # Default to first available method
            combined_onsets = list(onset_frames.values())[0]
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(combined_onsets, 
                                           sr=self.sr, 
                                           hop_length=self.hop_length)
        
        return np.array(sorted(onset_times))
    
    def _onset_spectral_flux(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Spectral flux based onset detection."""
        # Compute STFT
        stft = librosa.stft(audio, 
                           hop_length=self.hop_length, 
                           n_fft=self.frame_length)
        magnitude = np.abs(stft)
        
        # Spectral flux
        flux = np.sum(np.diff(magnitude, axis=1), axis=0)
        flux = np.maximum(0, flux)  # Half-wave rectification
        
        # Peak picking
        onset_frames = librosa.util.peak_pick(
            flux,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.1,
            wait=10
        )
        
        return onset_frames, flux
    
    def _onset_complex_domain(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Complex domain onset detection."""
        onset_envelope = librosa.onset.onset_strength(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            feature=librosa.feature.chroma_cqt
        )
        
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.sr,
            hop_length=self.hop_length,
            units='frames'
        )
        
        return onset_frames, onset_envelope
    
    def _onset_energy_based(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Energy-based onset detection."""
        # RMS energy
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
            frame_length=self.frame_length
        )[0]
        
        # Energy flux
        energy_flux = np.diff(rms)
        energy_flux = np.maximum(0, energy_flux)  # Half-wave rectification
        
        # Peak picking
        onset_frames = librosa.util.peak_pick(
            energy_flux,
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=3,
            delta=0.05,
            wait=5
        )
        
        return onset_frames, energy_flux
    
    def _onset_phase_deviation(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Phase deviation onset detection."""
        # Compute STFT
        stft = librosa.stft(audio, 
                           hop_length=self.hop_length, 
                           n_fft=self.frame_length)
        
        # Phase deviation
        phase = np.angle(stft)
        phase_dev = np.sum(np.abs(np.diff(np.unwrap(phase, axis=1), axis=1)), axis=0)
        
        # Peak picking
        onset_frames = librosa.util.peak_pick(
            phase_dev,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.1,
            wait=8
        )
        
        return onset_frames, phase_dev
    
    def _onset_high_frequency(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """High-frequency content onset detection."""
        # High-pass filter
        nyquist = self.sr // 2
        high_cutoff = 2000  # Hz
        b, a = scipy.signal.butter(4, high_cutoff / nyquist, btype='high')
        audio_hf = scipy.signal.filtfilt(b, a, audio)
        
        # Energy in high-frequency band
        hf_energy = librosa.feature.rms(
            y=audio_hf,
            hop_length=self.hop_length,
            frame_length=self.frame_length
        )[0]
        
        # Energy flux
        hf_flux = np.diff(hf_energy)
        hf_flux = np.maximum(0, hf_flux)
        
        # Peak picking
        onset_frames = librosa.util.peak_pick(
            hf_flux,
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=3,
            delta=0.03,
            wait=5
        )
        
        return onset_frames, hf_flux
    
    def _combine_onsets_weighted(self, 
                                onset_frames: Dict[str, np.ndarray],
                                onset_strengths: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine onsets using weighted voting."""
        # Weights for different methods (based on empirical performance)
        weights = {
            'spectral_flux': 0.3,
            'complex_domain': 0.25,
            'energy_based': 0.2,
            'phase_deviation': 0.15,
            'high_frequency': 0.1
        }
        
        # Create time grid
        max_frames = max(len(strength) for strength in onset_strengths.values())
        time_grid = np.arange(max_frames)
        
        # Weighted combination
        combined_strength = np.zeros(max_frames)
        total_weight = 0
        
        for method, frames in onset_frames.items():
            if method in weights and method in onset_strengths:
                weight = weights[method]
                strength = onset_strengths[method]
                
                # Pad or truncate to match grid
                if len(strength) < max_frames:
                    strength = np.pad(strength, (0, max_frames - len(strength)))
                elif len(strength) > max_frames:
                    strength = strength[:max_frames]
                
                combined_strength += weight * strength
                total_weight += weight
        
        if total_weight > 0:
            combined_strength /= total_weight
        
        # Peak picking on combined strength
        combined_onsets = librosa.util.peak_pick(
            combined_strength,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.1,
            wait=10
        )
        
        return combined_onsets
    
    def _combine_onsets_union(self, onset_frames: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine onsets using union (all detected onsets)."""
        all_onsets = []
        for frames in onset_frames.values():
            all_onsets.extend(frames)
        
        # Remove duplicates and sort
        unique_onsets = np.unique(all_onsets)
        
        # Merge nearby onsets (within 50ms)
        merge_threshold = int(0.05 * self.sr / self.hop_length)  # 50ms in frames
        merged_onsets = []
        
        for onset in unique_onsets:
            if not merged_onsets or onset - merged_onsets[-1] > merge_threshold:
                merged_onsets.append(onset)
        
        return np.array(merged_onsets)
    
    def _combine_onsets_intersection(self, onset_frames: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine onsets using intersection (only onsets detected by multiple methods)."""
        if len(onset_frames) < 2:
            return list(onset_frames.values())[0] if onset_frames else np.array([])
        
        # Convert to sets for intersection
        onset_sets = []
        tolerance = int(0.02 * self.sr / self.hop_length)  # 20ms tolerance in frames
        
        for frames in onset_frames.values():
            # Create ranges around each onset
            onset_ranges = []
            for frame in frames:
                onset_ranges.extend(range(max(0, frame - tolerance), frame + tolerance + 1))
            onset_sets.append(set(onset_ranges))
        
        # Find intersection
        common_frames = onset_sets[0]
        for onset_set in onset_sets[1:]:
            common_frames = common_frames.intersection(onset_set)
        
        # Convert back to onset frames (take center of ranges)
        if not common_frames:
            return np.array([])
        
        sorted_frames = sorted(common_frames)
        onset_frames_result = []
        
        i = 0
        while i < len(sorted_frames):
            start = sorted_frames[i]
            # Find end of consecutive range
            end = start
            while i + 1 < len(sorted_frames) and sorted_frames[i + 1] == sorted_frames[i] + 1:
                i += 1
                end = sorted_frames[i]
            
            # Take center of range
            center = (start + end) // 2
            onset_frames_result.append(center)
            i += 1
        
        return np.array(onset_frames_result)
    
    def detect_note_frames(self, 
                          audio: np.ndarray,
                          onsets: np.ndarray,
                          min_note_duration: float = 0.05) -> List[Tuple[float, float]]:
        """
        Detect note frames (start and end times) based on onsets.
        
        Args:
            audio: Audio signal
            onsets: Onset times in seconds
            min_note_duration: Minimum note duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each note
        """
        if len(onsets) == 0:
            return []
        
        # Compute RMS energy for offset detection
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
            frame_length=self.frame_length
        )[0]
        
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        note_frames = []
        
        for i, onset in enumerate(onsets):
            start_time = onset
            
            # Find offset (next onset or energy decay)
            if i + 1 < len(onsets):
                next_onset = onsets[i + 1]
                
                # Look for energy decay between onsets
                onset_idx = np.searchsorted(rms_times, onset)
                next_onset_idx = np.searchsorted(rms_times, next_onset)
                
                if onset_idx < len(rms) and next_onset_idx > onset_idx:
                    # Find minimum energy between onsets
                    energy_segment = rms[onset_idx:next_onset_idx]
                    if len(energy_segment) > 0:
                        min_energy_idx = np.argmin(energy_segment) + onset_idx
                        potential_offset = rms_times[min_energy_idx]
                        
                        # Use offset if it's reasonable, otherwise use next onset
                        if potential_offset > start_time + min_note_duration:
                            end_time = min(potential_offset, next_onset - 0.01)  # Small gap
                        else:
                            end_time = next_onset - 0.01
                    else:
                        end_time = next_onset - 0.01
                else:
                    end_time = next_onset - 0.01
            else:
                # Last note - use energy decay or fixed duration
                onset_idx = np.searchsorted(rms_times, onset)
                if onset_idx < len(rms):
                    # Look for significant energy decay
                    energy_after_onset = rms[onset_idx:]
                    if len(energy_after_onset) > 10:  # Need some frames to analyze
                        peak_energy = np.max(energy_after_onset[:5])  # Peak in first few frames
                        decay_threshold = peak_energy * 0.1  # 10% of peak
                        
                        decay_indices = np.where(energy_after_onset < decay_threshold)[0]
                        if len(decay_indices) > 0:
                            decay_idx = decay_indices[0] + onset_idx
                            end_time = rms_times[min(decay_idx, len(rms_times) - 1)]
                        else:
                            end_time = rms_times[-1]  # End of audio
                    else:
                        end_time = rms_times[-1]
                else:
                    end_time = len(audio) / self.sr
            
            # Ensure minimum duration
            if end_time - start_time < min_note_duration:
                end_time = start_time + min_note_duration
            
            note_frames.append((start_time, end_time))
        
        return note_frames
    
    def analyze_performance(self, audio: np.ndarray) -> Dict:
        """
        Complete performance analysis with onsets and frames.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting onset and frame detection analysis")
        
        # Detect onsets
        onsets = self.detect_onsets_multi_method(audio, combine_method='weighted')
        logger.info(f"Detected {len(onsets)} onsets")
        
        # Detect note frames
        note_frames = self.detect_note_frames(audio, onsets)
        logger.info(f"Detected {len(note_frames)} note frames")
        
        # Additional analysis
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Compute spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )[0]
        
        # RMS energy
        rms_energy = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length
        )[0]
        
        return {
            'onsets': onsets.tolist(),
            'note_frames': note_frames,
            'tempo': float(tempo),
            'beats': librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length).tolist(),
            'spectral_features': {
                'centroids': spectral_centroids.tolist(),
                'rolloff': spectral_rolloff.tolist(),
                'rms_energy': rms_energy.tolist()
            },
            'analysis_params': {
                'sr': self.sr,
                'hop_length': self.hop_length,
                'frame_length': self.frame_length,
                'frame_time': self.frame_time
            }
        }


def detect_onsets_and_frames(audio_path: str, 
                           sr: int = 22050,
                           method: str = 'bytedance') -> Dict:
    """
    Professional onset and frame detection with ByteDance Piano Transcription.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (ignored for ByteDance, uses optimal 16kHz)
        method: Detection method ('bytedance', 'librosa', 'basic_pitch')
        
    Returns:
        Analysis results dictionary
    """
    logger.info(f"Starting onset detection for {audio_path} with method: {method}")
    
    # Method 1: ByteDance Piano Transcription (BEST - Primary choice)
    if method == 'bytedance' and BYTEDANCE_AVAILABLE:
        try:
            logger.info("Using ByteDance Piano Transcription (superior chord detection)")
            result = analyze_piano_performance(audio_path)
            
            # Convert to expected format
            return {
                'onsets': result['onsets'],
                'note_frames': [(note['onset_s'], note['offset_s']) for note in result['notes']],
                'tempo': result['tempo'],
                'beats': result['beats'],
                'notes': result['notes'],  # Full note information
                'analysis_method': 'bytedance_piano_transcription',
                'chord_analysis': result.get('chord_analysis', {}),
                'transcription_quality': result.get('transcription_quality', {})
            }
            
        except Exception as e:
            logger.warning(f"ByteDance transcription failed: {e}, falling back to librosa")
            method = 'librosa'  # Fallback
    
    # Method 2: Multi-method librosa (GOOD - Reliable fallback)
    if method == 'librosa' or (method == 'bytedance' and not BYTEDANCE_AVAILABLE):
        try:
            logger.info("Using multi-method librosa onset detection")
            
            # Load audio
            audio, actual_sr = librosa.load(audio_path, sr=sr)
            
            # Create detector
            detector = OnsetFrameDetector(sr=actual_sr)
            
            # Analyze with multiple methods
            results = detector.analyze_performance(audio)
            results['analysis_method'] = 'librosa_multi_method'
            
            return results
            
        except Exception as e:
            logger.warning(f"Librosa detection failed: {e}, falling back to Basic Pitch")
            method = 'basic_pitch'  # Final fallback
    
    # Method 3: Basic Pitch (LEGACY - Last resort only)
    if method == 'basic_pitch' and BASIC_PITCH_AVAILABLE:
        try:
            logger.info("Using Basic Pitch (legacy fallback)")
            
            # Load audio for basic analysis
            audio, actual_sr = librosa.load(audio_path, sr=sr)
            
            # Basic Pitch transcription
            result = predict(audio_path)
            
            # Extract basic information
            if len(result) >= 2:
                midi_data, note_events = result[:2]
                
                # Convert to basic format
                onsets = []
                note_frames = []
                
                if hasattr(note_events, 'notes'):
                    for note in note_events.notes:
                        onsets.append(note.start)
                        note_frames.append((note.start, note.end))
                
                # Estimate tempo from onsets
                if len(onsets) > 1:
                    intervals = np.diff(sorted(onsets))
                    tempo = 60.0 / np.median(intervals) if len(intervals) > 0 else 120.0
                else:
                    tempo = 120.0
                
                return {
                    'onsets': onsets,
                    'note_frames': note_frames,
                    'tempo': tempo,
                    'beats': onsets,  # Approximate
                    'analysis_method': 'basic_pitch_fallback'
                }
            
        except Exception as e:
            logger.warning(f"Basic Pitch failed: {e}, using minimal librosa")
    
    # Final fallback: Minimal librosa onset detection
    logger.warning("All advanced methods failed, using basic librosa onset detection")
    
    try:
        audio, actual_sr = librosa.load(audio_path, sr=sr)
        
        # Basic onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=actual_sr, 
            hop_length=512,
            units='time'
        )
        
        # Basic tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=actual_sr)
        beat_times = librosa.frames_to_time(beats, sr=actual_sr, hop_length=512)
        
        return {
            'onsets': onset_frames.tolist(),
            'note_frames': [(onset, onset + 0.5) for onset in onset_frames],  # Assume 0.5s duration
            'tempo': float(tempo),
            'beats': beat_times.tolist(),
            'analysis_method': 'librosa_basic_fallback'
        }
        
    except Exception as e:
        logger.error(f"All onset detection methods failed: {e}")
        
        # Return minimal valid result
        return {
            'onsets': [],
            'note_frames': [],
            'tempo': 120.0,
            'beats': [0.0],
            'analysis_method': 'failed_fallback'
        }


if __name__ == "__main__":
    # Test the onset detection system
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        results = detect_onsets_and_frames(audio_path)
        
        print(f"Analysis Results for {audio_path}:")
        print(f"- Detected {len(results['onsets'])} onsets")
        print(f"- Detected {len(results['note_frames'])} note frames")
        print(f"- Estimated tempo: {results['tempo']:.1f} BPM")
        print(f"- Beat times: {len(results['beats'])} beats")
    else:
        print("Usage: python onset_detection.py <audio_file>")
