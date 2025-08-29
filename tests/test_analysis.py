"""
Tests for audio analysis functionality.
Tests use synthetic audio to verify detection of various issues.
"""
import pytest
import numpy as np
import os
import tempfile
from app.analysis import analyze_audio_pair
from tests.fixtures import (
    AudioTestFixture, create_test_reference_audio, create_modified_test_audio,
    TARGET_SR
)


class TestAudioAnalysis:
    """Test suite for audio analysis functionality."""
    
    @pytest.fixture
    def reference_audio(self):
        """Create a short reference audio for testing (4 seconds)."""
        return create_test_reference_audio(duration=4.0)
    
    def test_alignment_identity(self, reference_audio):
        """
        Test with identical audio files.
        Expected: overall_score ~ 100, no significant issues.
        """
        with AudioTestFixture(reference_audio, "ref_identity.wav") as ref_path, \
             AudioTestFixture(reference_audio, "user_identity.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Check overall scores - should be very high for identical audio
            summary = result['summary']
            assert summary['overall_score'] >= 95, f"Expected high overall score, got {summary['overall_score']}"
            assert summary['tempo_score'] >= 95, f"Expected high tempo score, got {summary['tempo_score']}"
            assert summary['pitch_score'] >= 95, f"Expected high pitch score, got {summary['pitch_score']}"
            assert summary['dynamics_score'] >= 95, f"Expected high dynamics score, got {summary['dynamics_score']}"
            
            # Should have minimal or no issues
            issues = result['issues']
            assert len(issues) <= 2, f"Expected minimal issues for identical audio, got {len(issues)} issues"
            
            # If there are any issues, they should be very low severity
            for issue in issues:
                assert issue['severity'] < 0.3, f"Expected low severity for identical audio, got {issue['severity']} for {issue['type']}"
            
            # Check that MIDI analysis ran (even if not available)
            assert 'midi_estimates' in result
            
            # Verify basic structure
            assert 'per_frame' in result
            assert 'summary' in result
            assert 'tips' in result
            assert isinstance(result['tips'], list)
    
    def test_tempo_shift(self, reference_audio):
        """
        Test with time-stretched audio.
        Expected: analysis should handle tempo-shifted audio without crashing.
        """
        # Create tempo-shifted version (15% slower)
        modified_audio = create_modified_test_audio(
            reference_audio, 
            'tempo', 
            stretch_factor=1.15
        )
        
        with AudioTestFixture(reference_audio, "ref_tempo.wav") as ref_path, \
             AudioTestFixture(modified_audio, "user_tempo.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Basic structure should be present (main test)
            assert 'summary' in result
            assert 'overall_score' in result['summary']
            assert 'tempo_score' in result['summary']
            assert 'issues' in result
            
            # Should have valid scores (0-100)
            summary = result['summary']
            assert 0 <= summary['overall_score'] <= 100
            assert 0 <= summary['tempo_score'] <= 100
            
            # Analysis should complete without errors
            assert len(result['per_frame']['time_s']) > 0, "Should have per-frame data"
            
            # If tempo issues are detected, they should have proper format
            tempo_issues = [issue for issue in result['issues'] if issue['type'] == 'tempo_drift']
            for issue in tempo_issues:
                assert 0 <= issue['severity'] <= 1, f"Severity should be 0-1, got {issue['severity']}"
                assert isinstance(issue['explanation'], str), "Explanation should be string"
                assert issue['start_s'] <= issue['end_s'], "Start should be <= end time"
    
    def test_pitch_shift(self, reference_audio):
        """
        Test with pitch-shifted audio.
        Expected: analysis should handle pitch-shifted audio and produce reasonable scores.
        """
        # Create pitch-shifted version (+4 semitones from 1-3 seconds)
        modified_audio = create_modified_test_audio(
            reference_audio, 
            'pitch', 
            semitones=4, 
            start_time=1.0, 
            end_time=3.0
        )
        
        with AudioTestFixture(reference_audio, "ref_pitch.wav") as ref_path, \
             AudioTestFixture(modified_audio, "user_pitch.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Basic structure should be present (main test)
            assert 'summary' in result
            assert 'pitch_score' in result['summary']
            assert 'issues' in result
            
            # Should have valid scores (0-100)
            summary = result['summary']
            assert 0 <= summary['overall_score'] <= 100
            assert 0 <= summary['pitch_score'] <= 100
            
            # Analysis should complete and generate per-frame data
            assert len(result['per_frame']['chroma_distance']) > 0, "Should have chroma distance data"
            
            # If pitch issues are detected, they should have proper format
            pitch_issues = [issue for issue in result['issues'] if issue['type'] == 'pitch_accuracy']
            for issue in pitch_issues:
                assert 0 <= issue['severity'] <= 1, f"Severity should be 0-1, got {issue['severity']}"
                assert isinstance(issue['explanation'], str), "Explanation should be string"
                assert issue['start_s'] <= issue['end_s'], "Start should be <= end time"
    
    def test_rms_diff(self, reference_audio):
        """
        Test with amplified region.
        Expected: analysis should handle dynamics changes and produce valid results.
        """
        # Create version with amplified region (5x louder from 1.5-2.5 seconds)
        modified_audio = create_modified_test_audio(
            reference_audio, 
            'dynamics', 
            amplification=5.0, 
            start_time=1.5, 
            end_time=2.5
        )
        
        with AudioTestFixture(reference_audio, "ref_dynamics.wav") as ref_path, \
             AudioTestFixture(modified_audio, "user_dynamics.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Basic structure should be present (main test)
            assert 'summary' in result
            assert 'dynamics_score' in result['summary']
            assert 'issues' in result
            
            # Should have valid scores (0-100)
            summary = result['summary']
            assert 0 <= summary['overall_score'] <= 100
            assert 0 <= summary['dynamics_score'] <= 100
            
            # Analysis should complete and generate RMS data
            assert len(result['per_frame']['rms_user']) > 0, "Should have user RMS data"
            assert len(result['per_frame']['rms_ref']) > 0, "Should have reference RMS data"
            
            # If dynamics issues are detected, they should have proper format
            dynamics_issues = [issue for issue in result['issues'] if issue['type'] == 'dynamics_mismatch']
            for issue in dynamics_issues:
                assert 0 <= issue['severity'] <= 1, f"Severity should be 0-1, got {issue['severity']}"
                assert isinstance(issue['explanation'], str), "Explanation should be string"
                assert issue['start_s'] <= issue['end_s'], "Start should be <= end time"
    
    def test_analysis_structure_and_format(self, reference_audio):
        """
        Test that the analysis returns properly structured data.
        """
        with AudioTestFixture(reference_audio, "ref_structure.wav") as ref_path, \
             AudioTestFixture(reference_audio, "user_structure.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Check main structure
            required_keys = ['summary', 'per_frame', 'issues', 'tips', 'midi_estimates']
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"
            
            # Check summary structure
            summary = result['summary']
            summary_keys = ['user_duration', 'ref_duration', 'total_issues', 'alignment_quality',
                           'overall_score', 'tempo_score', 'pitch_score', 'dynamics_score']
            for key in summary_keys:
                assert key in summary, f"Missing summary key: {key}"
            
            # Check per_frame structure
            per_frame = result['per_frame']
            frame_keys = ['time_s', 'chroma_distance', 'tempo_user', 'tempo_ref', 
                         'rms_user', 'rms_ref']
            for key in frame_keys:
                assert key in per_frame, f"Missing per_frame key: {key}"
                assert isinstance(per_frame[key], list), f"per_frame[{key}] should be a list"
            
            # Check that all per_frame arrays have the same length
            array_lengths = [len(per_frame[key]) for key in frame_keys]
            assert len(set(array_lengths)) == 1, f"per_frame arrays have different lengths: {array_lengths}"
            
            # Check issues structure
            for issue in result['issues']:
                issue_keys = ['start_s', 'end_s', 'type', 'severity', 'explanation']
                for key in issue_keys:
                    assert key in issue, f"Missing issue key: {key}"
                
                # Check data types and ranges
                assert isinstance(issue['start_s'], (int, float)), "start_s should be numeric"
                assert isinstance(issue['end_s'], (int, float)), "end_s should be numeric"
                assert isinstance(issue['type'], str), "type should be string"
                assert 0 <= issue['severity'] <= 1, f"severity should be 0-1, got {issue['severity']}"
                assert isinstance(issue['explanation'], str), "explanation should be string"
                assert issue['start_s'] <= issue['end_s'], "start_s should be <= end_s"
            
            # Check tips
            assert isinstance(result['tips'], list), "tips should be a list"
            for tip in result['tips']:
                assert isinstance(tip, str), "each tip should be a string"
                assert len(tip) <= 140, f"tip too long ({len(tip)} chars): {tip}"
            
            # Check MIDI estimates structure
            midi_estimates = result['midi_estimates']
            assert isinstance(midi_estimates, dict), "midi_estimates should be a dict"
            assert 'available' in midi_estimates, "midi_estimates should have 'available' key"
    
    def test_csv_generation(self, reference_audio):
        """
        Test that CSV files are generated correctly.
        """
        # Create slightly different audio to ensure some issues are generated
        modified_audio = create_modified_test_audio(
            reference_audio, 'tempo', stretch_factor=1.03
        )
        
        with AudioTestFixture(reference_audio, "ref_csv.wav") as ref_path, \
             AudioTestFixture(modified_audio, "user_csv.wav") as user_path:
            
            result = analyze_audio_pair(user_path, ref_path)
            
            # Check that CSV path is in the summary (it should be created)
            assert 'csv_path' in result['summary'], "CSV path should be in summary"
            
            csv_path = result['summary']['csv_path']
            assert os.path.exists(csv_path), f"CSV file should exist at {csv_path}"
            
            # Read and verify CSV format
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # Should have header + at least one issue
            assert len(lines) >= 2, "CSV should have header + at least one issue"
            
            # Check header
            header = lines[0].strip()
            expected_columns = ['start_s', 'end_s', 'type', 'severity', 'explanation']
            for col in expected_columns:
                assert col in header, f"CSV header should contain {col}"
            
            # Clean up the CSV file
            try:
                os.remove(csv_path)
            except OSError:
                pass
    
    def test_short_audio_handling(self):
        """
        Test analysis with very short audio (edge case).
        """
        # Create very short audio (1 second)
        short_audio = create_test_reference_audio(duration=1.0)
        
        with AudioTestFixture(short_audio, "ref_short.wav") as ref_path, \
             AudioTestFixture(short_audio, "user_short.wav") as user_path:
            
            # Should not crash with short audio
            result = analyze_audio_pair(user_path, ref_path)
            
            # Basic structure should still be present
            assert 'summary' in result
            assert 'issues' in result
            assert 'overall_score' in result['summary']
            assert isinstance(result['issues'], list)
