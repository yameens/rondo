#!/usr/bin/env python3
"""
Comprehensive test suite for ByteDance Piano Transcription integration.
Tests the complete pipeline to ensure everything works correctly.
"""

import os
import sys
import numpy as np
import soundfile as sf
import tempfile
import logging
from pathlib import Path
from typing import Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ByteDanceIntegrationTest:
    """Comprehensive test suite for ByteDance integration."""
    
    def __init__(self):
        self.test_results = []
        self.temp_files = []
        
    def log_test(self, test_name: str, success: bool, message: str, details=None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        
        if details and not success:
            logger.error(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'details': details
        })
        
        return success
    
    def create_test_piano_audio(self, 
                               duration: float = 5.0, 
                               sr: int = 16000,
                               notes: List[int] = None) -> str:
        """
        Create synthetic piano audio for testing.
        
        Args:
            duration: Audio duration in seconds
            sr: Sample rate
            notes: MIDI note numbers to synthesize
            
        Returns:
            Path to temporary audio file
        """
        if notes is None:
            # Create a simple C major scale + chord
            notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        
        # Generate time array
        t = np.linspace(0, duration, int(sr * duration), False)
        audio = np.zeros_like(t)
        
        # Add each note with different timing
        note_duration = duration / len(notes)
        
        for i, midi_note in enumerate(notes):
            # Convert MIDI to frequency
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            
            # Note timing
            start_time = i * note_duration
            end_time = (i + 1) * note_duration
            
            # Create note envelope
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            
            if end_idx > len(t):
                end_idx = len(t)
            
            note_length = end_idx - start_idx
            if note_length > 0:
                # Generate note with envelope
                note_t = np.linspace(0, note_duration, note_length, False)
                envelope = np.exp(-3 * note_t / note_duration)  # Exponential decay
                note_signal = 0.3 * envelope * np.sin(2 * np.pi * freq * note_t)
                
                # Add harmonics for more realistic piano sound
                note_signal += 0.1 * envelope * np.sin(2 * np.pi * freq * 2 * note_t)  # Octave
                note_signal += 0.05 * envelope * np.sin(2 * np.pi * freq * 3 * note_t)  # Fifth
                
                audio[start_idx:end_idx] += note_signal
        
        # Add a chord at the end for chord detection testing
        chord_start = int(0.8 * len(audio))
        chord_notes = [60, 64, 67]  # C major triad
        
        for midi_note in chord_notes:
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            chord_length = len(audio) - chord_start
            chord_t = np.linspace(0, chord_length / sr, chord_length, False)
            envelope = np.exp(-2 * chord_t / (chord_length / sr))
            chord_signal = 0.2 * envelope * np.sin(2 * np.pi * freq * chord_t)
            audio[chord_start:] += chord_signal
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio, sr)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        logger.info(f"Created test audio: {temp_file.name} ({duration}s, {len(notes)} notes + chord)")
        
        return temp_file.name
    
    def test_bytedance_import(self):
        """Test ByteDance module import."""
        try:
            from backend.app.piano_transcription import (
                transcribe_piano_audio,
                analyze_piano_performance,
                test_bytedance_integration
            )
            return self.log_test("ByteDance Import", True, "All modules imported successfully")
        except ImportError as e:
            return self.log_test("ByteDance Import", False, f"Import failed: {e}")
    
    def test_bytedance_initialization(self):
        """Test ByteDance transcriber initialization."""
        try:
            from backend.app.piano_transcription import test_bytedance_integration
            
            success = test_bytedance_integration()
            
            if success:
                return self.log_test("ByteDance Init", True, "Transcriber initialized successfully")
            else:
                return self.log_test("ByteDance Init", False, "Transcriber initialization failed")
                
        except Exception as e:
            return self.log_test("ByteDance Init", False, f"Initialization error: {e}")
    
    def test_audio_transcription(self):
        """Test audio transcription with synthetic piano audio."""
        try:
            from backend.app.piano_transcription import transcribe_piano_audio
            
            # Create test audio
            audio_path = self.create_test_piano_audio(duration=3.0)
            
            # Transcribe
            notes = transcribe_piano_audio(audio_path, confidence_threshold=0.05)
            
            if len(notes) > 0:
                # Analyze results
                note_count = len(notes)
                avg_confidence = np.mean([n['confidence'] for n in notes])
                tempo_estimate = 60.0 / np.mean(np.diff([n['onset_s'] for n in notes])) if len(notes) > 1 else 0
                
                return self.log_test("Audio Transcription", True, 
                                   f"Transcribed {note_count} notes, avg confidence: {avg_confidence:.2f}, tempo: {tempo_estimate:.1f} BPM")
            else:
                return self.log_test("Audio Transcription", False, "No notes detected in test audio")
                
        except Exception as e:
            return self.log_test("Audio Transcription", False, f"Transcription failed: {e}")
    
    def test_chord_detection(self):
        """Test chord detection capabilities (ByteDance's strength)."""
        try:
            from backend.app.piano_transcription import transcribe_piano_audio, analyze_chord_density
            
            # Create audio with simultaneous notes (chord)
            chord_notes = [60, 64, 67, 72]  # C major with octave
            audio_path = self.create_test_piano_audio(duration=2.0, notes=chord_notes)
            
            # Transcribe
            notes = transcribe_piano_audio(audio_path, confidence_threshold=0.05)
            
            if len(notes) > 0:
                # Analyze chord detection
                chord_analysis = analyze_chord_density(notes, simultaneity_threshold=0.1)
                
                polyphonic_ratio = chord_analysis['polyphonic_ratio']
                max_simultaneous = chord_analysis['max_simultaneous_notes']
                
                if polyphonic_ratio > 0.3 or max_simultaneous > 2:
                    return self.log_test("Chord Detection", True, 
                                       f"Detected chords: {polyphonic_ratio:.1%} polyphonic, max {max_simultaneous} simultaneous notes")
                else:
                    return self.log_test("Chord Detection", False, 
                                       f"Poor chord detection: {polyphonic_ratio:.1%} polyphonic, max {max_simultaneous} simultaneous")
            else:
                return self.log_test("Chord Detection", False, "No notes detected for chord test")
                
        except Exception as e:
            return self.log_test("Chord Detection", False, f"Chord detection test failed: {e}")
    
    def test_onset_detection_integration(self):
        """Test integration with onset detection system."""
        try:
            from backend.app.onset_detection import detect_onsets_and_frames
            
            # Create test audio
            audio_path = self.create_test_piano_audio(duration=4.0)
            
            # Test ByteDance method
            result = detect_onsets_and_frames(audio_path, method='bytedance')
            
            if result['analysis_method'] == 'bytedance_piano_transcription':
                onset_count = len(result['onsets'])
                tempo = result['tempo']
                
                return self.log_test("Onset Integration", True, 
                                   f"ByteDance integration working: {onset_count} onsets, {tempo:.1f} BPM")
            else:
                return self.log_test("Onset Integration", False, 
                                   f"ByteDance not used, fallback method: {result['analysis_method']}")
                
        except Exception as e:
            return self.log_test("Onset Integration", False, f"Integration test failed: {e}")
    
    def test_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        try:
            from backend.app.analysis import analyze_audio_pair
            
            # Create student and reference audio
            student_path = self.create_test_piano_audio(duration=3.0, notes=[60, 62, 64, 65])
            reference_path = self.create_test_piano_audio(duration=3.0, notes=[60, 62, 64, 67])
            
            # Run complete analysis
            result = analyze_audio_pair(student_path, reference_path)
            
            if 'student' in result and 'reference' in result:
                student_method = result['student'].get('analysis_method', 'unknown')
                reference_method = result['reference']['onset_analysis'].get('analysis_method', 'unknown')
                
                student_notes = result['student']['num_notes']
                reference_notes = result['reference']['num_notes']
                
                return self.log_test("Analysis Pipeline", True, 
                                   f"Complete analysis: student ({student_method}, {student_notes} notes), reference ({reference_method}, {reference_notes} notes)")
            else:
                return self.log_test("Analysis Pipeline", False, "Incomplete analysis result")
                
        except Exception as e:
            return self.log_test("Analysis Pipeline", False, f"Pipeline test failed: {e}")
    
    def test_fallback_system(self):
        """Test fallback system when ByteDance fails."""
        try:
            from backend.app.onset_detection import detect_onsets_and_frames
            
            # Create test audio
            audio_path = self.create_test_piano_audio(duration=2.0)
            
            # Test librosa fallback
            result = detect_onsets_and_frames(audio_path, method='librosa')
            
            if result['analysis_method'] in ['librosa_multi_method', 'librosa_basic_fallback']:
                onset_count = len(result['onsets'])
                return self.log_test("Fallback System", True, 
                                   f"Fallback working: {result['analysis_method']}, {onset_count} onsets")
            else:
                return self.log_test("Fallback System", False, 
                                   f"Unexpected fallback method: {result['analysis_method']}")
                
        except Exception as e:
            return self.log_test("Fallback System", False, f"Fallback test failed: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🎹 STARTING BYTEDANCE INTEGRATION TESTS")
        logger.info("=" * 60)
        
        tests = [
            ("ByteDance Import", self.test_bytedance_import),
            ("ByteDance Initialization", self.test_bytedance_initialization),
            ("Audio Transcription", self.test_audio_transcription),
            ("Chord Detection", self.test_chord_detection),
            ("Onset Integration", self.test_onset_detection_integration),
            ("Analysis Pipeline", self.test_analysis_pipeline),
            ("Fallback System", self.test_fallback_system),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_test(test_name, False, f"Test crashed: {e}")
        
        # Generate report
        self.generate_report(passed, total)
        
        return passed == total
    
    def generate_report(self, passed: int, total: int):
        """Generate final test report."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 BYTEDANCE INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # Categorize results
        by_status = {}
        for result in self.test_results:
            status = "PASS" if result['success'] else "FAIL"
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(result)
        
        for status, results in by_status.items():
            logger.info(f"\n{status}: {len(results)} tests")
            for result in results:
                logger.info(f"  - {result['test']}: {result['message']}")
        
        # Final assessment
        logger.info(f"\n🎯 INTEGRATION ASSESSMENT:")
        if passed == total:
            logger.info("  🟢 BYTEDANCE INTEGRATION FULLY FUNCTIONAL")
            logger.info("  🟢 SUPERIOR CHORD DETECTION READY")
            logger.info("  🟢 READY FOR PRODUCTION USE")
        elif passed >= total * 0.8:
            logger.info("  🟡 MOSTLY FUNCTIONAL - Minor issues")
        else:
            logger.info("  🔴 INTEGRATION ISSUES - Needs attention")


def main():
    """Main test function."""
    tester = ByteDanceIntegrationTest()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit(main())
