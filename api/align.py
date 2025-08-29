import numpy as np
import librosa

def pitch_to_chroma(midi):
    """
    Convert MIDI pitch to 12-dimensional chroma vector.
    
    Args:
        midi: MIDI pitch number
        
    Returns:
        12-dimensional chroma vector
    """
    pc = midi % 12
    v = np.zeros(12, dtype=float)
    v[pc] = 1.0
    return v

def score_beatchroma_from_events(events, beat_grid):
    """
    Build score chromagram from note events.
    
    Args:
        events: List of note events
        beat_grid: List of global beat centers
        
    Returns:
        Score chromagram (12 x num_beats)
    """
    chroma = np.zeros((12, len(beat_grid)))
    
    for i, beat in enumerate(beat_grid):
        # Find notes that are sounding at this beat
        active_notes = []
        for event in events:
            onset = event["onset_beat"]
            duration = event["duration_beats"]
            
            # Check if note is active at this beat
            if onset <= beat < onset + duration:
                active_notes.append(event["pitch_midi"])
        
        if active_notes:
            # Sum chroma vectors for all active notes
            v = np.zeros(12)
            for pitch in active_notes:
                v += pitch_to_chroma(pitch)
            
            # Normalize
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            
            chroma[:, i] = v
    
    return chroma

def dtw_align(score_chroma, audio_chroma):
    """
    Perform DTW alignment between score and audio chroma.
    
    Args:
        score_chroma: Score chromagram (12 x num_score_beats)
        audio_chroma: Audio chromagram (12 x num_audio_beats)
        
    Returns:
        Mapping from score beat index to audio beat index, and distance matrix
    """
    # Use cosine distance for chroma comparison
    D, wp = librosa.sequence.dtw(X=score_chroma, Y=audio_chroma, metric='cosine')
    
    # Reverse path to get forward order
    wp = wp[::-1]
    
    # Create mapping from score beat index to audio beat index
    mapping = {}
    for i, j in wp:
        mapping[int(i)] = int(j)
    
    return mapping, D

def match_notes(score_events, audio_data, dtw_mapping, tolerance=0.2):
    """
    Match score notes to audio onsets.
    
    Args:
        score_events: List of score note events
        audio_data: Audio features dictionary
        dtw_mapping: DTW mapping from score to audio beats
        tolerance: Timing tolerance in seconds
        
    Returns:
        Dictionary with matched, missed, and extra notes
    """
    matched_notes = []
    missed_notes = []
    extra_notes = []
    
    # Get audio onset times
    audio_onsets = audio_data["onset_times"]
    
    # For each score note
    for event in score_events:
        score_beat = event["onset_beat"]
        pitch_midi = event["pitch_midi"]
        
        # Map score beat to audio time via DTW
        if score_beat in dtw_mapping:
            audio_beat_idx = dtw_mapping[score_beat]
            
            # Convert beat index to time (approximate)
            if audio_beat_idx < len(audio_data["beats_sec"]):
                expected_time = audio_data["beats_sec"][audio_beat_idx]
                
                # Find nearby audio onsets
                nearby_onsets = []
                for onset_time in audio_onsets:
                    if abs(onset_time - expected_time) <= tolerance:
                        nearby_onsets.append(onset_time)
                
                if nearby_onsets:
                    # Check if pitch is present in audio chroma at this time
                    audio_beat_chroma = audio_data["beat_chroma"][:, audio_beat_idx]
                    expected_pc = pitch_midi % 12
                    
                    if audio_beat_chroma[expected_pc] > 0.1:  # Threshold for "note present"
                        matched_notes.append({
                            "score_event": event,
                            "audio_time": nearby_onsets[0],
                            "confidence": audio_beat_chroma[expected_pc]
                        })
                    else:
                        missed_notes.append(event)
                else:
                    missed_notes.append(event)
            else:
                missed_notes.append(event)
        else:
            missed_notes.append(event)
    
    # Find extra notes (audio onsets not matched to score)
    matched_times = [note["audio_time"] for note in matched_notes]
    
    for onset_time in audio_onsets:
        # Check if this onset is close to any matched note
        is_matched = False
        for matched_time in matched_times:
            if abs(onset_time - matched_time) <= tolerance:
                is_matched = True
                break
        
        if not is_matched:
            # This is an extra note
            extra_notes.append({
                "audio_time": onset_time,
                "type": "extra"
            })
    
    return {
        "matched": matched_notes,
        "missed": missed_notes,
        "extra": extra_notes
    }

def calculate_tempo_ratios(dtw_mapping, score_beats, audio_beats):
    """
    Calculate tempo ratios per measure from DTW mapping.
    
    Args:
        dtw_mapping: DTW mapping from score to audio beats
        score_beats: List of score beat times
        audio_beats: List of audio beat times
        
    Returns:
        Dictionary mapping measure index to tempo ratio
    """
    tempo_ratios = {}
    
    # Group beats by measure (assuming 4/4 time)
    beats_per_measure = 4
    
    for measure_idx in range(1, len(score_beats) // beats_per_measure + 1):
        start_beat = (measure_idx - 1) * beats_per_measure
        end_beat = measure_idx * beats_per_measure
        
        # Get score beats for this measure
        measure_score_beats = [i for i in range(start_beat, end_beat) if i in dtw_mapping]
        
        if len(measure_score_beats) >= 2:
            # Get corresponding audio beats
            measure_audio_beats = [dtw_mapping[i] for i in measure_score_beats]
            
            # Calculate tempo ratio
            score_duration = len(measure_score_beats) - 1  # In beats
            audio_duration = measure_audio_beats[-1] - measure_audio_beats[0]  # In audio beat indices
            
            if score_duration > 0 and audio_duration > 0:
                tempo_ratio = audio_duration / score_duration
                tempo_ratios[measure_idx] = tempo_ratio
    
    return tempo_ratios

def analyze_dynamics_deviation(score_dynamics, audio_dynamics, dtw_mapping):
    """
    Analyze dynamics deviation between score and audio.
    
    Args:
        score_dynamics: Score dynamic markings
        audio_dynamics: Audio RMS values per measure
        dtw_mapping: DTW mapping
        
    Returns:
        Dynamics deviation per measure
    """
    dynamics_deviation = {}
    
    # Map dynamics markings to expected RMS ranges
    dynamic_ranges = {
        "pp": (0.0, 0.2),
        "p": (0.2, 0.4),
        "mp": (0.4, 0.6),
        "mf": (0.6, 0.8),
        "f": (0.8, 0.9),
        "ff": (0.9, 1.0)
    }
    
    for measure_idx, audio_rms in audio_dynamics.items():
        # Find score dynamics for this measure
        measure_score_dynamics = [
            d for d in score_dynamics 
            if d.get("measure") == measure_idx
        ]
        
        if measure_score_dynamics and audio_rms:
            # Get expected RMS range
            dynamic_mark = measure_score_dynamics[0]["value"]
            expected_range = dynamic_ranges.get(dynamic_mark, (0.5, 0.7))
            
            # Calculate deviation
            avg_rms = audio_rms["avg_rms"]
            if avg_rms < expected_range[0]:
                deviation = (avg_rms - expected_range[0]) / expected_range[0]
            elif avg_rms > expected_range[1]:
                deviation = (avg_rms - expected_range[1]) / expected_range[1]
            else:
                deviation = 0.0
            
            dynamics_deviation[measure_idx] = deviation
    
    return dynamics_deviation
