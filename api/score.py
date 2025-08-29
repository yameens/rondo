from music21 import converter, note, chord, dynamics, tempo
import os

def parse_score_musicxml(path):
    """
    Parse MusicXML file and extract note events, dynamics, and tempo marks.
    
    Args:
        path: Path to MusicXML file
        
    Returns:
        Dictionary with events, dynamic_marks, and tempo_marks
    """
    try:
        # Parse the score
        score = converter.parse(path)
        
        # Get the first part (assuming solo piano)
        if len(score.parts) == 0:
            raise ValueError("No parts found in score")
        
        part = score.parts[0]
        
        # Extract note events
        events = []
        current_measure = 1
        
        for measure in part.getElementsByClass('Measure'):
            measure_number = measure.number if measure.number else current_measure
            
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    events.append({
                        "pitch_midi": element.pitch.midi,
                        "onset_beat": float(element.offset),  # Local to measure
                        "duration_beats": float(element.quarterLength),
                        "measure_index": measure_number,
                        "velocity": 64,  # Default velocity
                    })
                elif isinstance(element, chord.Chord):
                    # Handle chords by creating separate events for each note
                    for pitch in element.pitches:
                        events.append({
                            "pitch_midi": pitch.midi,
                            "onset_beat": float(element.offset),
                            "duration_beats": float(element.quarterLength),
                            "measure_index": measure_number,
                            "velocity": 64,
                        })
            
            current_measure += 1
        
        # Extract dynamic marks
        dynamic_marks = []
        for dynamic in score.recurse().getElementsByClass(dynamics.Dynamic):
            dynamic_marks.append({
                "offset": float(dynamic.offset),
                "value": dynamic.value,
                "measure": dynamic.getContextByClass('Measure').number if dynamic.getContextByClass('Measure') else 1
            })
        
        # Extract tempo marks
        tempo_marks = []
        for tempo_mark in score.recurse().getElementsByClass(tempo.MetronomeMark):
            tempo_marks.append({
                "offset": float(tempo_mark.offset),
                "number": tempo_mark.number,
                "measure": tempo_mark.getContextByClass('Measure').number if tempo_mark.getContextByClass('Measure') else 1
            })
        
        # Get time signature info
        time_signatures = []
        for ts in score.recurse().getElementsByClass('TimeSignature'):
            time_signatures.append({
                "offset": float(ts.offset),
                "numerator": ts.numerator,
                "denominator": ts.denominator,
                "measure": ts.getContextByClass('Measure').number if ts.getContextByClass('Measure') else 1
            })
        
        # Get key signature info
        key_signatures = []
        for ks in score.recurse().getElementsByClass('KeySignature'):
            key_signatures.append({
                "offset": float(ks.offset),
                "sharps": ks.sharps,
                "measure": ks.getContextByClass('Measure').number if ks.getContextByClass('Measure') else 1
            })
        
        return {
            "events": events,
            "dynamic_marks": dynamic_marks,
            "tempo_marks": tempo_marks,
            "time_signatures": time_signatures,
            "key_signatures": key_signatures,
            "total_measures": current_measure - 1
        }
        
    except Exception as e:
        print(f"Error parsing score: {str(e)}")
        raise ValueError(f"Failed to parse MusicXML file: {str(e)}")

def convert_global_beats(events, time_signatures):
    """
    Convert local measure offsets to global beat positions.
    This is a simplified version for MVP - assumes 4/4 time.
    
    Args:
        events: List of note events with local offsets
        time_signatures: List of time signature changes
        
    Returns:
        Events with global beat positions
    """
    # For MVP, assume 4/4 time throughout
    # In a full implementation, you'd track time signature changes
    beats_per_measure = 4
    
    for event in events:
        measure_idx = event["measure_index"]
        local_offset = event["onset_beat"]
        # Convert to global beats (measure 1 starts at beat 0)
        global_beat = (measure_idx - 1) * beats_per_measure + local_offset
        event["global_beat"] = global_beat
    
    return events

def get_measure_boundaries(events, time_signatures):
    """
    Get the start and end beats for each measure.
    
    Args:
        events: List of note events
        time_signatures: List of time signature changes
        
    Returns:
        Dictionary mapping measure number to (start_beat, end_beat)
    """
    # For MVP, assume 4/4 time
    beats_per_measure = 4
    measure_boundaries = {}
    
    max_measure = max(event["measure_index"] for event in events)
    
    for measure_idx in range(1, max_measure + 1):
        start_beat = (measure_idx - 1) * beats_per_measure
        end_beat = measure_idx * beats_per_measure
        measure_boundaries[measure_idx] = (start_beat, end_beat)
    
    return measure_boundaries
