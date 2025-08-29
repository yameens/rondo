import librosa
import numpy as np
import soundfile as sf

def load_audio_feats(path, sr=44100):
    """
    Load audio file and extract features for analysis.
    
    Args:
        path: Path to audio file
        sr: Sample rate (default 44100)
        
    Returns:
        Dictionary with audio features
    """
    try:
        # Load audio
        y, sr = librosa.load(path, sr=sr, mono=True)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Chroma features (frame-level)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
        
        # Map frames to nearest beat index
        beat_indices = np.searchsorted(times, beats, side='left')
        
        # Aggregate chroma per beat
        beat_chroma = []
        last_idx = 0
        
        for idx in beat_indices:
            if idx > last_idx:
                segment = chroma[:, last_idx:idx]
                beat_chroma.append(np.mean(segment, axis=1))
            else:
                # If no frames between beats, use single frame
                segment = chroma[:, last_idx:last_idx+1]
                beat_chroma.append(np.mean(segment, axis=1))
            last_idx = idx
        
        # Handle remaining frames
        if last_idx < chroma.shape[1]:
            segment = chroma[:, last_idx:]
            beat_chroma.append(np.mean(segment, axis=1))
        
        beat_chroma = np.stack(beat_chroma, axis=1) if beat_chroma else chroma
        
        # RMS for dynamics analysis
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        
        # Aggregate RMS per beat
        beat_rms = []
        last_idx = 0
        
        for idx in beat_indices:
            if idx < len(rms_times):
                rms_idx = np.searchsorted(rms_times, beats[idx], side='left')
                if rms_idx > last_idx:
                    segment = rms[last_idx:rms_idx]
                    beat_rms.append(np.mean(segment))
                else:
                    beat_rms.append(rms[last_idx] if last_idx < len(rms) else 0)
                last_idx = rms_idx
        
        # Handle remaining RMS frames
        if last_idx < len(rms):
            segment = rms[last_idx:]
            beat_rms.append(np.mean(segment))
        
        beat_rms = np.array(beat_rms) if beat_rms else np.array([0])
        
        # Spectral centroid for timbre analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Zero crossing rate for onset detection
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            "tempo": tempo,
            "beats_sec": beats,
            "beat_chroma": beat_chroma,
            "beat_rms": beat_rms,
            "onset_times": onset_times,
            "rms": rms,
            "rms_times": rms_times,
            "spectral_centroid": spectral_centroid,
            "zcr": zcr,
            "sr": sr,
            "y": y,
            "duration": len(y) / sr
        }
        
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        raise ValueError(f"Failed to analyze audio file: {str(e)}")

def detect_note_onsets(audio_data, threshold=0.1):
    """
    Detect individual note onsets from audio.
    
    Args:
        audio_data: Audio features dictionary
        threshold: Onset detection threshold
        
    Returns:
        List of onset times
    """
    # Use onset strength for more precise detection
    onset_strength = librosa.onset.onset_strength(
        y=audio_data["y"], 
        sr=audio_data["sr"]
    )
    
    onset_frames = librosa.onset.onset_detect(
        onset_strength=onset_strength,
        threshold=threshold,
        units='frames'
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=audio_data["sr"])
    
    return onset_times

def extract_pitch_contour(audio_data):
    """
    Extract pitch contour from audio.
    
    Args:
        audio_data: Audio features dictionary
        
    Returns:
        Pitch contour array
    """
    # Use pYIN for pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data["y"],
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=audio_data["sr"]
    )
    
    return f0, voiced_flag

def analyze_dynamics(audio_data, measure_boundaries):
    """
    Analyze dynamics per measure.
    
    Args:
        audio_data: Audio features dictionary
        measure_boundaries: Dictionary of measure boundaries
        
    Returns:
        Dynamics analysis per measure
    """
    dynamics_per_measure = {}
    
    for measure_idx, (start_beat, end_beat) in measure_boundaries.items():
        # Find beats within this measure
        measure_beats = []
        for i, beat_time in enumerate(audio_data["beats_sec"]):
            beat_idx = i  # Assuming beats are evenly spaced
            if start_beat <= beat_idx < end_beat:
                measure_beats.append(i)
        
        if measure_beats:
            # Get RMS values for this measure
            measure_rms = [audio_data["beat_rms"][i] for i in measure_beats if i < len(audio_data["beat_rms"])]
            
            if measure_rms:
                avg_rms = np.mean(measure_rms)
                max_rms = np.max(measure_rms)
                min_rms = np.min(measure_rms)
                
                dynamics_per_measure[measure_idx] = {
                    "avg_rms": avg_rms,
                    "max_rms": max_rms,
                    "min_rms": min_rms,
                    "dynamic_range": max_rms - min_rms
                }
    
    return dynamics_per_measure
