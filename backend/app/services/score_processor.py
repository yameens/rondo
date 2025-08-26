import music21
import tempfile
import os
import subprocess
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET
from pathlib import Path
import zipfile
import io


class ScoreProcessor:
    def __init__(self):
        self.supported_formats = ['.xml', '.musicxml', '.mxl', '.mei', '.pdf']
    
    def process_score(self, file_path: str) -> Dict[str, Any]:
        """
        Process score file and return parsed MusicXML data.
        
        Args:
            file_path: Path to the score file
            
        Returns:
            Dictionary containing parsed score data and metadata
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Use OMR to convert PDF to MusicXML
            musicxml_path = self._convert_pdf_to_musicxml(file_path)
        elif file_ext == '.mxl':
            # Extract MusicXML from compressed MXL
            musicxml_path = self._extract_mxl(file_path)
        else:
            # Direct MusicXML/MEI processing
            musicxml_path = file_path
        
        # Parse MusicXML
        score_data = self._parse_musicxml(musicxml_path)
        
        # Clean up temporary files
        if file_ext in ['.pdf', '.mxl'] and musicxml_path != file_path:
            os.unlink(musicxml_path)
        
        return score_data
    
    def _convert_pdf_to_musicxml(self, pdf_path: str) -> str:
        """Convert PDF to MusicXML using Audiveris OMR."""
        # For local development, we'll use a simpler approach
        # In production, this would use the Audiveris Docker container
        
        # Check if we're in a Docker environment
        audiveris_paths = [
            "/app/Audiveris-5.6.2/bin/Audiveris-5.6.2.jar",  # Docker path
            "/usr/local/bin/audiveris",  # System install
            "audiveris"  # PATH install
        ]
        
        audiveris_cmd = None
        for path in audiveris_paths:
            if os.path.exists(path):
                audiveris_cmd = path
                break
        
        if not audiveris_cmd:
            # For now, create a more realistic placeholder MusicXML
            # In a real deployment, you'd want to install Audiveris or use the Docker container
            print("Warning: Audiveris not found. Creating placeholder MusicXML.")
            placeholder_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>480</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
      </attributes>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>D</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>E</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>F</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
    </measure>
    <measure number="2">
      <note>
        <pitch>
          <step>G</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>A</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>B</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>C</step>
          <octave>5</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
    </measure>
    <measure number="3">
      <note>
        <pitch>
          <step>C</step>
          <octave>5</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>B</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>A</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>G</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
    </measure>
    <measure number="4">
      <note>
        <pitch>
          <step>F</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>E</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>D</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>480</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>'''
            
            output_path = tempfile.mktemp(suffix='.xml')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(placeholder_xml)
            return output_path
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run Audiveris OMR
            if audiveris_cmd.endswith('.jar'):
                cmd = [
                    "java", "-Xmx4g", "-jar", audiveris_cmd,
                    "-batch", "-transcribe", "-export",
                    "-output", temp_dir,
                    pdf_path
                ]
            else:
                cmd = [
                    audiveris_cmd,
                    "-batch", "-transcribe", "-export",
                    "-output", temp_dir,
                    pdf_path
                ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                result.check_returncode()
            except subprocess.TimeoutExpired:
                raise Exception("OMR processing timed out")
            except subprocess.CalledProcessError as e:
                raise Exception(f"OMR processing failed: {e.stderr}")
            
            # Find the generated MusicXML file
            musicxml_files = list(Path(temp_dir).glob("*.xml"))
            if not musicxml_files:
                raise Exception("No MusicXML file generated by OMR")
            
            # Copy to a new temporary file (since temp_dir will be deleted)
            output_path = tempfile.mktemp(suffix='.xml')
            import shutil
            shutil.copy2(musicxml_files[0], output_path)
            
            return output_path
    
    def _extract_mxl(self, mxl_path: str) -> str:
        """Extract MusicXML from compressed MXL file."""
        with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
            # First, check if there's a container.xml file
            container_files = [f for f in zip_ref.namelist() if 'container.xml' in f]
            
            if container_files:
                # Read the container to find the main score file
                container_content = zip_ref.read(container_files[0])
                try:
                    import xml.etree.ElementTree as ET
                    container_root = ET.fromstring(container_content)
                    rootfile_elements = container_root.findall('.//rootfile')
                    if rootfile_elements:
                        # Get the path to the main score file
                        main_score_path = rootfile_elements[0].get('full-path')
                        if main_score_path and main_score_path in zip_ref.namelist():
                            xml_content = zip_ref.read(main_score_path)
                        else:
                            raise Exception(f"Main score file {main_score_path} not found in archive")
                    else:
                        raise Exception("No rootfile found in container.xml")
                except Exception as e:
                    print(f"Warning: Could not parse container.xml: {e}")
                    # Fall back to finding any XML file
                    xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml') and 'container' not in f]
                    if not xml_files:
                        raise Exception("No MusicXML file found in MXL archive")
                    xml_content = zip_ref.read(xml_files[0])
            else:
                # No container.xml, just find any XML file
                xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml')]
                if not xml_files:
                    raise Exception("No MusicXML file found in MXL archive")
                xml_content = zip_ref.read(xml_files[0])
            
            # Write to temporary file
            output_path = tempfile.mktemp(suffix='.xml')
            with open(output_path, 'wb') as f:
                f.write(xml_content)
            
            return output_path
    
    def _parse_musicxml(self, musicxml_path: str) -> Dict[str, Any]:
        """Parse MusicXML file and extract structured data."""
        try:
            # Load with music21
            score = music21.converter.parse(musicxml_path)
        except Exception as e:
            # Try to fix common issues
            print(f"Initial parsing failed: {e}")
            
            # Read the file and try to fix XML structure
            try:
                with open(musicxml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(musicxml_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Try to extract score-partwise if it's wrapped in container
            if 'container' in content and 'score-partwise' in content:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(content)
                    score_elements = root.findall('.//*[local-name()="score-partwise"]')
                    if score_elements:
                        score_content = ET.tostring(score_elements[0], encoding='unicode')
                        # Write fixed content to temporary file
                        fixed_path = tempfile.mktemp(suffix='.xml')
                        with open(fixed_path, 'w', encoding='utf-8') as f:
                            f.write(score_content)
                        score = music21.converter.parse(fixed_path)
                        # Clean up
                        os.unlink(fixed_path)
                    else:
                        # Try alternative approach - look for any score-like content
                        for elem in root.iter():
                            if elem.tag and 'score' in elem.tag.lower():
                                try:
                                    score_content = ET.tostring(elem, encoding='unicode')
                                    fixed_path = tempfile.mktemp(suffix='.xml')
                                    with open(fixed_path, 'w', encoding='utf-8') as f:
                                        f.write(score_content)
                                    score = music21.converter.parse(fixed_path)
                                    os.unlink(fixed_path)
                                    break
                                except:
                                    continue
                        else:
                            raise Exception("Could not extract score content from container")
                except Exception as e2:
                    print(f"Failed to fix XML structure: {e2}")
                    # Try one more approach - extract all XML content
                    try:
                        # Find the first XML declaration and extract everything after it
                        if '<?xml' in content:
                            start = content.find('<?xml')
                            xml_start = content.find('>', start) + 1
                            xml_content = content[xml_start:]
                            
                            # Look for score-partwise in the content
                            if '<score-partwise' in xml_content:
                                score_start = xml_content.find('<score-partwise')
                                score_end = xml_content.rfind('</score-partwise>') + len('</score-partwise>')
                                score_content = xml_content[score_start:score_end]
                                
                                fixed_path = tempfile.mktemp(suffix='.xml')
                                with open(fixed_path, 'w', encoding='utf-8') as f:
                                    f.write(score_content)
                                score = music21.converter.parse(fixed_path)
                                os.unlink(fixed_path)
                            else:
                                raise Exception("No score-partwise found in content")
                        else:
                            raise Exception("No XML declaration found")
                    except Exception as e3:
                        raise Exception(f"All parsing attempts failed: {e3}")
            else:
                raise e
        
        # Extract basic metadata
        metadata = {
            "title": score.metadata.title or "Untitled",
            "composer": score.metadata.composer or "Unknown",
            "number_of_parts": len(score.parts),
            "number_of_measures": len(score.recurse().getElementsByClass('Measure')),
            "time_signatures": [],
            "key_signatures": [],
            "tempo_markings": []
        }
        
        # Extract time signatures
        for ts in score.recurse().getElementsByClass('TimeSignature'):
            metadata["time_signatures"].append({
                "measure": ts.measureNumber,
                "numerator": ts.numerator,
                "denominator": ts.denominator
            })
        
        # Extract key signatures
        for ks in score.recurse().getElementsByClass('KeySignature'):
            metadata["key_signatures"].append({
                "measure": ks.measureNumber,
                "sharps": ks.sharps
            })
        
        # Extract tempo markings
        for tempo in score.recurse().getElementsByClass('MetronomeMark'):
            metadata["tempo_markings"].append({
                "measure": tempo.measureNumber,
                "number": tempo.number,
                "text": str(tempo)
            })
        
        # Extract notes and measures
        measures_data = []
        for part in score.parts:
            part_data = {
                "part_name": part.partName or f"Part {part.id}",
                "part_id": part.id,
                "measures": []
            }
            
            for measure in part.getElementsByClass('Measure'):
                measure_data = {
                    "measure_number": measure.number,
                    "notes": [],
                    "rests": [],
                    "chords": []
                }
                
                # Extract notes
                for note in measure.recurse().getElementsByClass('Note'):
                    try:
                        note_data = {
                            "pitch": int(note.pitch.midi) if note.pitch.midi is not None else 60,
                            "pitch_name": note.pitch.nameWithOctave,
                            "duration_quarter_length": float(note.duration.quarterLength) if note.duration.quarterLength is not None else 1.0,
                            "onset_quarter": float(note.offset) if note.offset is not None else 0.0,
                            "velocity": 80,  # Default velocity
                            "tie": note.tie is not None
                        }
                        measure_data["notes"].append(note_data)
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Warning: Skipping malformed note {note}: {e}")
                        continue
                
                # Extract rests
                for rest in measure.recurse().getElementsByClass('Rest'):
                    try:
                        rest_data = {
                            "duration_quarter_length": float(rest.duration.quarterLength) if rest.duration.quarterLength is not None else 1.0,
                            "onset_quarter": float(rest.offset) if rest.offset is not None else 0.0
                        }
                        measure_data["rests"].append(rest_data)
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Warning: Skipping malformed rest {rest}: {e}")
                        continue
                
                # Extract chords
                for chord in measure.recurse().getElementsByClass('Chord'):
                    try:
                        chord_data = {
                            "pitches": [int(p.midi) if p.midi is not None else 60 for p in chord.pitches],
                            "pitch_names": [p.nameWithOctave for p in chord.pitches],
                            "duration_quarter_length": float(chord.duration.quarterLength) if chord.duration.quarterLength is not None else 1.0,
                            "onset_quarter": float(chord.offset) if chord.offset is not None else 0.0,
                            "velocity": 80
                        }
                        measure_data["chords"].append(chord_data)
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Warning: Skipping malformed chord {chord}: {e}")
                        continue
                
                part_data["measures"].append(measure_data)
            
            measures_data.append(part_data)
        
        return {
            "metadata": metadata,
            "parts": measures_data,
            "musicxml_path": musicxml_path
        }
    
    def convert_to_midi_reference(self, score_data: Dict[str, Any], tempo: float = 120.0) -> List[Dict[str, Any]]:
        """Convert score data to MIDI-like reference sequence for alignment."""
        reference_events = []
        
        # Convert quarter notes to seconds based on tempo
        quarter_to_seconds = 60.0 / tempo
        
        # Process ALL parts (not just the first one)
        for part in score_data["parts"]:
            current_time = 0.0  # Each part starts at time 0
            
            for measure in part["measures"]:
                # Process notes
                for note in measure["notes"]:
                    onset_s = current_time + (note["onset_quarter"] * quarter_to_seconds)
                    duration_s = note["duration_quarter_length"] * quarter_to_seconds
                    offset_s = onset_s + duration_s
                    
                    reference_events.append({
                        "pitch": note["pitch"],
                        "onset_s": onset_s,
                        "offset_s": offset_s,
                        "velocity": note["velocity"],
                        "duration_s": duration_s,
                        "measure": measure["measure_number"],
                        "part_id": part["part_id"],
                        "type": "note"
                    })
                
                # Process chords
                for chord in measure["chords"]:
                    onset_s = current_time + (chord["onset_quarter"] * quarter_to_seconds)
                    duration_s = chord["duration_quarter_length"] * quarter_to_seconds
                    offset_s = onset_s + duration_s
                    
                    for pitch in chord["pitches"]:
                        reference_events.append({
                            "pitch": pitch,
                            "onset_s": onset_s,
                            "offset_s": offset_s,
                            "velocity": chord["velocity"],
                            "duration_s": duration_s,
                            "measure": measure["measure_number"],
                            "part_id": part["part_id"],
                            "type": "chord_note"
                        })
                
                # Calculate actual measure duration from notes/rests/chords
                max_offset = 0.0
                for note in measure["notes"]:
                    note_end = note["onset_quarter"] + note["duration_quarter_length"]
                    max_offset = max(max_offset, note_end)
                for chord in measure["chords"]:
                    chord_end = chord["onset_quarter"] + chord["duration_quarter_length"]
                    max_offset = max(max_offset, chord_end)
                for rest in measure["rests"]:
                    rest_end = rest["onset_quarter"] + rest["duration_quarter_length"]
                    max_offset = max(max_offset, rest_end)
                
                # Use actual measure duration, fallback to 4.0 if no content
                measure_duration = max_offset if max_offset > 0 else 4.0
                current_time += measure_duration * quarter_to_seconds
        
        # Sort by onset time
        reference_events.sort(key=lambda x: x["onset_s"])
        
        return reference_events
