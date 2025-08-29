#!/usr/bin/env python3
"""
Unit tests for app.utils module.
"""

import os
import tempfile
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import (
    safe_tmp_path,
    standardize_audio_to_wav,
    verify_file_size,
    convert_to_wav_with_pydub,
    download_youtube_audio,
    process_youtube_url,
    cleanup_temp_file,
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_BYTES,
    TARGET_SAMPLE_RATE
)


class TestSafeTmpPath:
    """Test safe_tmp_path function for uniqueness and proper format."""
    
    def test_default_suffix(self):
        """Test default .wav suffix."""
        path = safe_tmp_path()
        assert path.endswith('.wav')
        assert 'audio_' in os.path.basename(path)
    
    def test_custom_suffix(self):
        """Test custom file suffix."""
        path = safe_tmp_path('.mp3')
        assert path.endswith('.mp3')
        assert 'audio_' in os.path.basename(path)
    
    def test_uniqueness(self):
        """Test that multiple calls generate unique paths."""
        paths = [safe_tmp_path() for _ in range(10)]
        assert len(set(paths)) == 10  # All paths should be unique
    
    def test_path_format(self):
        """Test that path is in temp directory and properly formatted."""
        path = safe_tmp_path()
        temp_dir = tempfile.gettempdir()
        assert path.startswith(temp_dir)
        assert os.path.basename(path).startswith('audio_')


class TestVerifyFileSize:
    """Test file size verification functionality."""
    
    def test_file_within_limit(self):
        """Test file within size limit passes verification."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write small amount of data
            tmp_file.write(b'small file content')
            tmp_file.flush()
            
            try:
                verify_file_size(tmp_file.name)  # Should not raise
            finally:
                os.unlink(tmp_file.name)
    
    def test_file_exceeds_limit(self):
        """Test file exceeding size limit raises HTTPException."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write more than MAX_FILE_SIZE_BYTES
            large_data = b'x' * (MAX_FILE_SIZE_BYTES + 1)
            tmp_file.write(large_data)
            tmp_file.flush()
            
            try:
                with pytest.raises(HTTPException) as exc_info:
                    verify_file_size(tmp_file.name)
                assert exc_info.value.status_code == 413
                assert "exceeds maximum" in str(exc_info.value.detail)
            finally:
                os.unlink(tmp_file.name)
    
    def test_nonexistent_file(self):
        """Test that nonexistent file doesn't raise exception."""
        verify_file_size('/nonexistent/file/path')  # Should not raise


class TestStandardizeAudioToWav:
    """Test audio standardization functionality."""
    
    def create_test_audio_file(self, duration=1.0, sample_rate=44100, file_format='wav'):
        """Helper to create test audio files."""
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name
    
    def test_wav_round_trip(self):
        """Test that WAV files can be processed and maintain expected properties."""
        input_file = self.create_test_audio_file(duration=2.0, sample_rate=44100)
        
        try:
            output_path, duration = standardize_audio_to_wav(input_file)
            
            # Verify output file exists and has correct properties
            assert os.path.exists(output_path)
            
            # Load the output file to verify properties
            audio_data, sr = sf.read(output_path)
            assert sr == TARGET_SAMPLE_RATE
            assert len(audio_data.shape) == 1  # Mono
            assert abs(duration - 2.0) < 0.1  # Duration should be close to 2 seconds
            
            # Clean up
            cleanup_temp_file(output_path)
        finally:
            cleanup_temp_file(input_file)
    
    def test_sample_rate_conversion(self):
        """Test that audio is properly resampled to target sample rate."""
        # Create audio at different sample rate
        input_file = self.create_test_audio_file(duration=1.0, sample_rate=48000)
        
        try:
            output_path, duration = standardize_audio_to_wav(input_file)
            
            # Verify the output has the target sample rate
            audio_data, sr = sf.read(output_path)
            assert sr == TARGET_SAMPLE_RATE
            
            cleanup_temp_file(output_path)
        finally:
            cleanup_temp_file(input_file)
    
    def test_custom_sample_rate(self):
        """Test that custom sample rate parameter works."""
        input_file = self.create_test_audio_file()
        custom_rate = 16000
        
        try:
            output_path, duration = standardize_audio_to_wav(input_file, sample_rate=custom_rate)
            
            # Verify the output has the custom sample rate
            audio_data, sr = sf.read(output_path)
            assert sr == custom_rate
            
            cleanup_temp_file(output_path)
        finally:
            cleanup_temp_file(input_file)
    
    def test_duration_limit_enforcement(self):
        """Test that audio exceeding duration limit raises HTTPException."""
        # Create a short audio file first, then patch librosa to return long duration
        input_file = self.create_test_audio_file(duration=1.0)  # Create small file
        
        try:
            with patch('app.utils.librosa.get_duration', return_value=MAX_DURATION_SECONDS + 10):
                with pytest.raises(HTTPException) as exc_info:
                    standardize_audio_to_wav(input_file)
                
                assert exc_info.value.status_code == 413
                assert "duration" in str(exc_info.value.detail)
                assert "exceeds maximum" in str(exc_info.value.detail)
        finally:
            cleanup_temp_file(input_file)
    
    def test_custom_output_path(self):
        """Test that custom output path is respected."""
        input_file = self.create_test_audio_file()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
            custom_output = output_file.name
        
        try:
            output_path, duration = standardize_audio_to_wav(input_file, output_path=custom_output)
            assert output_path == custom_output
            assert os.path.exists(custom_output)
        finally:
            cleanup_temp_file(input_file)
            cleanup_temp_file(custom_output)
    
    @patch('app.utils.convert_to_wav_with_pydub')
    def test_non_wav_conversion(self, mock_convert):
        """Test that non-WAV files trigger conversion."""
        # Create a mock mp3 file (just for the extension check)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
            # Write some dummy data
            mp3_file.write(b'dummy mp3 data')
            mp3_file.flush()
            
            # Mock the conversion to create a proper WAV file
            def mock_convert_side_effect(input_path, output_path):
                # Create a real WAV file for the subsequent processing
                test_wav = self.create_test_audio_file()
                with open(test_wav, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                cleanup_temp_file(test_wav)
            
            mock_convert.side_effect = mock_convert_side_effect
            
            try:
                output_path, duration = standardize_audio_to_wav(mp3_file.name)
                
                # Verify conversion was called
                mock_convert.assert_called_once()
                assert os.path.exists(output_path)
                
                cleanup_temp_file(output_path)
            finally:
                cleanup_temp_file(mp3_file.name)


class TestConvertToWavWithPydub:
    """Test pydub audio conversion functionality."""
    
    @patch('app.utils.AudioSegment')
    def test_successful_conversion(self, mock_audio_segment):
        """Test successful audio conversion with pydub."""
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        
        convert_to_wav_with_pydub('input.mp3', 'output.wav')
        
        mock_audio_segment.from_file.assert_called_once_with('input.mp3')
        mock_audio.export.assert_called_once_with('output.wav', format='wav')
    
    @patch('app.utils.AudioSegment')
    def test_conversion_failure(self, mock_audio_segment):
        """Test that conversion errors are properly handled."""
        mock_audio_segment.from_file.side_effect = Exception("Conversion failed")
        
        with pytest.raises(HTTPException) as exc_info:
            convert_to_wav_with_pydub('input.mp3', 'output.wav')
        
        assert exc_info.value.status_code == 500
        assert "conversion to WAV failed" in str(exc_info.value.detail)


class TestDownloadYoutubeAudio:
    """Test YouTube audio download functionality."""
    
    @patch('app.utils.yt_dlp.YoutubeDL')
    def test_successful_download(self, mock_ydl_class):
        """Test successful YouTube audio download."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        # Mock video info with acceptable duration
        mock_info = {'duration': 120}  # 2 minutes
        mock_ydl.extract_info.return_value = mock_info
        
        # Mock that the file exists after download
        with patch('os.path.exists', return_value=True):
            result = download_youtube_audio('https://youtube.com/watch?v=test')
            
            mock_ydl.extract_info.assert_called()
            mock_ydl.download.assert_called_once()
            assert result is not None
    
    @patch('app.utils.yt_dlp.YoutubeDL')
    def test_duration_limit_exceeded(self, mock_ydl_class):
        """Test that videos exceeding duration limit are rejected."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        # Mock video info with excessive duration
        mock_info = {'duration': MAX_DURATION_SECONDS + 100}
        mock_ydl.extract_info.return_value = mock_info
        
        with pytest.raises(HTTPException) as exc_info:
            download_youtube_audio('https://youtube.com/watch?v=test')
        
        assert exc_info.value.status_code == 413
        assert "duration" in str(exc_info.value.detail)
        
        # Verify that download was not called since duration check failed
        mock_ydl.download.assert_not_called()
    
    @patch('app.utils.yt_dlp.YoutubeDL')
    def test_download_error(self, mock_ydl_class):
        """Test handling of download errors."""
        import yt_dlp
        
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        mock_ydl.extract_info.side_effect = yt_dlp.DownloadError("Download failed")
        
        with pytest.raises(HTTPException) as exc_info:
            download_youtube_audio('https://youtube.com/watch?v=test')
        
        assert exc_info.value.status_code == 400
        assert "YouTube download failed" in str(exc_info.value.detail)


class TestProcessYoutubeUrl:
    """Test complete YouTube processing pipeline."""
    
    @patch('app.utils.download_youtube_audio')
    @patch('app.utils.standardize_audio_to_wav')
    def test_successful_pipeline(self, mock_standardize, mock_download):
        """Test successful YouTube URL processing."""
        # Mock download returning a path
        mock_download.return_value = '/tmp/downloaded.webm'
        
        # Mock standardization returning final result
        mock_standardize.return_value = ('/tmp/final.wav', 120.5)
        
        with patch('os.path.exists', return_value=True), \
             patch('os.unlink') as mock_unlink:
            
            result_path, duration = process_youtube_url('https://youtube.com/watch?v=test')
            
            assert result_path == '/tmp/final.wav'
            assert duration == 120.5
            
            # Verify cleanup was called
            mock_unlink.assert_called()
    
    @patch('app.utils.download_youtube_audio')
    def test_pipeline_cleanup_on_error(self, mock_download):
        """Test that temporary files are cleaned up on error."""
        # Mock download returning a path
        mock_download.return_value = '/tmp/downloaded.webm'
        
        # Mock standardization failing
        with patch('app.utils.standardize_audio_to_wav', side_effect=Exception("Processing failed")), \
             patch('os.path.exists', return_value=True), \
             patch('os.unlink') as mock_unlink:
            
            with pytest.raises(Exception):
                process_youtube_url('https://youtube.com/watch?v=test')
            
            # Verify cleanup was still called
            mock_unlink.assert_called()


class TestCleanupTempFile:
    """Test temporary file cleanup functionality."""
    
    def test_cleanup_existing_file(self):
        """Test cleanup of existing temporary file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b'test content')
        
        # File should exist
        assert os.path.exists(tmp_path)
        
        # Cleanup should remove it
        cleanup_temp_file(tmp_path)
        assert not os.path.exists(tmp_path)
    
    def test_cleanup_nonexistent_file(self):
        """Test that cleanup of nonexistent file doesn't raise error."""
        cleanup_temp_file('/nonexistent/file/path')  # Should not raise
    
    def test_cleanup_none_path(self):
        """Test that cleanup with None path doesn't raise error."""
        cleanup_temp_file(None)  # Should not raise
    
    def test_cleanup_empty_path(self):
        """Test that cleanup with empty path doesn't raise error."""
        cleanup_temp_file('')  # Should not raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
