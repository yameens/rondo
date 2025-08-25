import pytest
import numpy as np
from unittest.mock import Mock, patch
from backend.app.services.audio_processor import AudioProcessor


class TestAudioProcessor:
    @pytest.fixture
    def audio_processor(self):
        return AudioProcessor()
    
    @pytest.fixture
    def mock_audio_data(self):
        # Create mock audio data
        return np.random.rand(44100 * 5)  # 5 seconds of audio
    
    def test_validate_audio_duration_ok(self, audio_processor, tmp_path):
        """Test that valid audio duration passes validation."""
        # Create a mock audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"mock audio data")
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(44100 * 30), 44100)  # 30 seconds
            
            result = audio_processor._validate_audio(audio_file)
            
            assert result["original_duration"] == 30.0
            assert result["sample_rate"] == 44100
            assert result["channels"] == 1
    
    def test_validate_audio_duration_too_long(self, audio_processor, tmp_path):
        """Test that audio exceeding max duration raises error."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"mock audio data")
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(44100 * 700), 44100)  # 700 seconds
            
            with pytest.raises(ValueError, match="exceeds maximum allowed"):
                audio_processor._validate_audio(audio_file)
    
    def test_normalize_events(self, audio_processor):
        """Test event normalization."""
        mock_events = [
            (60, 1.0, 1.5, 80, 0.95),  # pitch, onset, offset, velocity, confidence
            (64, 2.0, 2.25, 90, 0.88),
        ]
        
        result = audio_processor._normalize_events(mock_events)
        
        assert len(result) == 2
        assert result[0]["pitch"] == 60
        assert result[0]["onset_s"] == 1.0
        assert result[0]["offset_s"] == 1.5
        assert result[0]["velocity"] == 80
        assert result[0]["confidence"] == 0.95
        assert result[0]["duration_s"] == 0.5
        assert result[0]["midi_note"] == 60
    
    def test_extract_features(self, audio_processor, tmp_path):
        """Test audio feature extraction."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"mock audio data")
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.beat.beat_track') as mock_tempo, \
             patch('librosa.onset.onset_strength') as mock_onset, \
             patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.feature.spectral_rolloff') as mock_rolloff:
            
            mock_load.return_value = (np.random.rand(44100 * 10), 44100)
            mock_tempo.return_value = (120.0, np.array([0, 1, 2]))
            mock_onset.return_value = np.array([0.1, 0.2, 0.3])
            mock_centroid.return_value = np.array([[0.5, 0.6, 0.7]])
            mock_rolloff.return_value = np.array([[0.8, 0.9, 1.0]])
            
            result = audio_processor.extract_features(audio_file)
            
            assert result["tempo"] == 120.0
            assert len(result["onset_strength"]) == 3
            assert len(result["spectral_centroids"]) == 3
            assert len(result["spectral_rolloff"]) == 3
