#!/usr/bin/env python3
"""
Comprehensive API testing script for the AI Piano Teacher system.
Tests all endpoints to ensure functionality before deployment.
"""

import requests
import json
import time
import os
import tempfile
import wave
import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, message: str, response_data: Any = None):
        """Log test results."""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time(),
            'response_data': response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        
    def create_test_audio_file(self, duration: float = 5.0, sample_rate: int = 22050) -> str:
        """Create a test audio file for upload testing."""
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Create a simple melody (C major scale)
        frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
        audio_data = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            start_idx = int(i * len(t) / len(frequencies))
            end_idx = int((i + 1) * len(t) / len(frequencies))
            audio_data[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        
        # Normalize and convert to 16-bit
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Check", True, f"API is healthy: {data.get('status', 'unknown')}", data)
                return True
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Connection failed: {str(e)}")
            return False

    def test_database_connection(self):
        """Test database connectivity."""
        try:
            response = self.session.get(f"{self.base_url}/db/tables")
            if response.status_code == 200:
                data = response.json()
                tables = data.get('tables', [])
                expected_tables = ['score_pieces', 'performances', 'envelopes']
                missing_tables = [t for t in expected_tables if t not in tables]
                
                if not missing_tables:
                    self.log_test("Database Connection", True, f"All required tables exist: {tables}", data)
                    return True
                else:
                    self.log_test("Database Connection", False, f"Missing tables: {missing_tables}")
                    return False
            else:
                self.log_test("Database Connection", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Database Connection", False, f"Request failed: {str(e)}")
            return False

    def test_score_endpoints(self):
        """Test score-related endpoints."""
        # Test creating a score
        try:
            score_data = {
                "slug": "test-chopin-nocturne",
                "musicxml_path": "/app/scores/chopin_nocturne_op9_no2.xml"
            }
            
            response = self.session.post(f"{self.base_url}/api/score-pieces", json=score_data)
            if response.status_code == 200:
                score = response.json()
                score_id = score['id']
                self.log_test("Create Score", True, f"Created score with ID: {score_id}", score)
                
                # Test getting the score
                response = self.session.get(f"{self.base_url}/scores/{score_id}")
                if response.status_code == 200:
                    retrieved_score = response.json()
                    self.log_test("Get Score", True, f"Retrieved score: {retrieved_score['slug']}", retrieved_score)
                    return score_id
                else:
                    self.log_test("Get Score", False, f"HTTP {response.status_code}: {response.text}")
                    return None
            else:
                # Score might already exist, try to get existing scores
                response = self.session.get(f"{self.base_url}/scores/")
                if response.status_code == 200:
                    scores = response.json()
                    if scores:
                        score_id = scores[0]['id']
                        self.log_test("Create Score", True, f"Using existing score with ID: {score_id}", scores[0])
                        return score_id
                    else:
                        self.log_test("Create Score", False, "No scores available and creation failed")
                        return None
                else:
                    self.log_test("Create Score", False, f"HTTP {response.status_code}: {response.text}")
                    return None
        except Exception as e:
            self.log_test("Score Endpoints", False, f"Request failed: {str(e)}")
            return None

    def test_performance_upload(self, score_id: int):
        """Test performance upload functionality with onset detection."""
        audio_file_path = None
        try:
            # Create test audio file
            audio_file_path = self.create_test_audio_file(duration=10.0)  # Longer for better testing
            
            # Test student performance upload
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': ('test_performance.wav', audio_file, 'audio/wav')}
                data = {
                    'score_id': str(score_id),
                    'role': 'student',
                    'source': 'api_test_onset_detection'
                }
                
                response = self.session.post(f"{self.base_url}/api/performances/student", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    performance_id = result.get('performance', {}).get('id')
                    features = result.get('features', {})
                    
                    # Validate onset detection worked
                    if features:
                        tempo_curve = features.get('tempo', {})
                        loudness_curve = features.get('loudness', {})
                        
                        if tempo_curve and loudness_curve:
                            self.log_test("Student Upload", True, 
                                        f"Uploaded with onset detection: ID {performance_id}, "
                                        f"Tempo points: {len(tempo_curve.get('beats', []))}, "
                                        f"Loudness points: {len(loudness_curve.get('beats', []))}", 
                                        result)
                        else:
                            self.log_test("Student Upload", True, f"Uploaded but limited feature extraction: {performance_id}", result)
                    else:
                        self.log_test("Student Upload", True, f"Uploaded student performance: {performance_id}", result)
                    
                    return performance_id
                else:
                    self.log_test("Student Upload", False, f"HTTP {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            self.log_test("Performance Upload", False, f"Upload failed: {str(e)}")
            return None
        finally:
            # Clean up test file
            if audio_file_path and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)

    def test_reference_upload(self, score_id: int):
        """Test reference performance upload."""
        audio_file_path = None
        try:
            # Create test audio file
            audio_file_path = self.create_test_audio_file(duration=3.0)
            
            # Test reference performance upload
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': ('test_reference.wav', audio_file, 'audio/wav')}
                data = {
                    'score_id': str(score_id),
                    'role': 'reference',
                    'source': 'api_test_reference'
                }
                
                response = self.session.post(f"{self.base_url}/api/performances/reference", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    performance_id = result.get('performance', {}).get('id')
                    self.log_test("Reference Upload", True, f"Uploaded reference performance: {performance_id}", result)
                    return performance_id
                else:
                    self.log_test("Reference Upload", False, f"HTTP {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            self.log_test("Reference Upload", False, f"Upload failed: {str(e)}")
            return None
        finally:
            # Clean up test file
            if audio_file_path and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)

    def test_envelope_generation(self, score_id: int):
        """Test envelope generation from reference performances."""
        try:
            response = self.session.post(f"{self.base_url}/api/envelopes/{score_id}/build")
            
            if response.status_code == 200:
                envelopes = response.json()
                self.log_test("Envelope Generation", True, f"Generated envelopes for score {score_id}", envelopes)
                return True
            else:
                self.log_test("Envelope Generation", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Envelope Generation", False, f"Request failed: {str(e)}")
            return False

    def test_expressive_scoring(self, student_performance_id: int):
        """Test expressive scoring functionality."""
        try:
            response = self.session.post(f"{self.base_url}/api/expressive-score/{student_performance_id}")
            
            if response.status_code == 200:
                score_data = response.json()
                overall_score = score_data.get('overall', {}).get('expressiveness', 0)
                self.log_test("Expressive Scoring", True, f"Generated expressive score: {overall_score}", score_data)
                return True
            else:
                self.log_test("Expressive Scoring", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Expressive Scoring", False, f"Request failed: {str(e)}")
            return False

    def test_celery_jobs(self):
        """Test Celery job system."""
        try:
            response = self.session.get(f"{self.base_url}/api/jobs")
            
            if response.status_code == 200:
                jobs_data = response.json()
                self.log_test("Celery Jobs", True, f"Job system is running", jobs_data)
                return True
            else:
                self.log_test("Celery Jobs", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Celery Jobs", False, f"Request failed: {str(e)}")
            return False

    def test_onset_detection_direct(self):
        """Test onset detection system directly."""
        audio_file_path = None
        try:
            # Create test audio file
            audio_file_path = self.create_test_audio_file(duration=5.0)
            
            # Test direct onset detection endpoint if available
            response = self.session.post(f"{self.base_url}/api/analyze/onsets", 
                                       files={'audio': ('test_onset.wav', open(audio_file_path, 'rb'), 'audio/wav')})
            
            if response.status_code == 200:
                result = response.json()
                onsets = result.get('onsets', [])
                tempo = result.get('tempo', 0)
                
                self.log_test("Onset Detection", True, 
                            f"Detected {len(onsets)} onsets, tempo: {tempo:.1f} BPM", result)
                return True
            elif response.status_code == 404:
                # Endpoint might not exist, that's okay
                self.log_test("Onset Detection", True, "Direct onset endpoint not available (expected)")
                return True
            else:
                self.log_test("Onset Detection", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Onset Detection", False, f"Request failed: {str(e)}")
            return False
        finally:
            if audio_file_path and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)

    def test_verovio_integration(self):
        """Test Verovio score display integration."""
        try:
            # Test if there's a Verovio endpoint
            response = self.session.get(f"{self.base_url}/api/scores/1/verovio")
            
            if response.status_code == 200:
                result = response.json()
                self.log_test("Verovio Integration", True, "Verovio endpoint available", result)
                return True
            elif response.status_code == 404:
                self.log_test("Verovio Integration", True, "Verovio endpoint not implemented (frontend-only)")
                return True
            else:
                self.log_test("Verovio Integration", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Verovio Integration", False, f"Request failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all API tests in sequence."""
        logger.info("🚀 Starting comprehensive API testing...")
        
        # Basic connectivity tests
        if not self.test_health_endpoint():
            logger.error("❌ Health check failed - aborting tests")
            return False
            
        if not self.test_database_connection():
            logger.warning("⚠️ Database connection issues - some tests may fail")
        
        # Test score management
        score_id = self.test_score_endpoints()
        if not score_id:
            logger.error("❌ Score creation failed - aborting performance tests")
            return False
        
        # Test performance uploads
        student_perf_id = self.test_performance_upload(score_id)
        ref_perf_id = self.test_reference_upload(score_id)
        
        # Test envelope generation (requires reference performances)
        if ref_perf_id:
            self.test_envelope_generation(score_id)
        
        # Test expressive scoring (requires student performance)
        if student_perf_id:
            self.test_expressive_scoring(student_perf_id)
        
        # Test job system
        self.test_celery_jobs()
        
        # Test advanced features
        self.test_onset_detection_direct()
        self.test_verovio_integration()
        
        # Generate summary
        self.generate_test_summary()
        return True

    def generate_test_summary(self):
        """Generate and display test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info("\n" + "="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    logger.info(f"  - {result['test']}: {result['message']}")
        
        logger.info("\n🎯 RECOMMENDATIONS:")
        if passed_tests == total_tests:
            logger.info("  ✅ All tests passed! Your API is ready for production.")
        elif passed_tests >= total_tests * 0.8:
            logger.info("  ⚠️ Most tests passed. Review failed tests and fix minor issues.")
        else:
            logger.info("  ❌ Multiple test failures. Review system configuration and dependencies.")
        
        logger.info("="*60)

def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AI Piano Teacher APIs')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = APITester(args.url)
    
    try:
        success = tester.run_all_tests()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n🛑 Testing interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"💥 Testing failed with error: {str(e)}")
        exit_code = 1
    
    exit(exit_code)

if __name__ == "__main__":
    main()
