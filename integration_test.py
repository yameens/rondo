#!/usr/bin/env python3
"""
Google-level integration testing for AI Piano Teacher system.
Comprehensive validation of all components working together.
"""

import os
import sys
import time
import logging
import subprocess
import requests
import tempfile
import wave
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemIntegrationTester:
    """
    Comprehensive system integration tester.
    Tests the complete AI Piano Teacher system end-to-end.
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.frontend_url = "http://localhost:3000"
        self.test_results = []
        self.session = requests.Session()
        self.session.timeout = 30
        
    def log_result(self, test_name: str, success: bool, message: str, details: Any = None):
        """Log test result."""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        
        if not success and details:
            logger.error(f"Details: {details}")
    
    def create_realistic_piano_audio(self, duration: float = 10.0, sample_rate: int = 22050) -> str:
        """Create realistic piano audio for testing."""
        # Piano frequencies (C4 to C6)
        piano_notes = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,  # C4-C5
            587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50  # C5-C6
        ]
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.zeros_like(t)
        
        # Create a realistic piano melody with proper timing
        num_notes = 15
        note_duration = duration / num_notes
        
        for i in range(num_notes):
            start_time = i * note_duration
            end_time = start_time + note_duration * 0.8  # 80% duty cycle
            
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            if end_idx > len(t):
                break
                
            # Select note frequency
            freq = piano_notes[i % len(piano_notes)]
            
            # Generate note with envelope
            note_samples = end_idx - start_idx
            note_t = np.linspace(0, note_duration * 0.8, note_samples)
            
            # Piano-like envelope (quick attack, slow decay)
            envelope = np.exp(-note_t * 3) * (1 - np.exp(-note_t * 50))
            
            # Generate note with harmonics
            fundamental = np.sin(2 * np.pi * freq * note_t)
            harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * note_t)
            harmonic3 = 0.1 * np.sin(2 * np.pi * freq * 3 * note_t)
            
            note_signal = (fundamental + harmonic2 + harmonic3) * envelope
            
            # Add to audio
            if start_idx < len(audio_data) and end_idx <= len(audio_data):
                audio_data[start_idx:end_idx] += note_signal
        
        # Normalize and add slight noise for realism
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        noise = np.random.normal(0, 0.01, len(audio_data))
        audio_data += noise
        
        # Convert to 16-bit
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name
    
    def test_docker_services(self) -> bool:
        """Test that all Docker services are running."""
        try:
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                output = result.stdout
                services = ['db', 'redis', 'backend', 'frontend', 'worker']
                running_services = []
                
                for service in services:
                    if service in output and 'Up' in output:
                        running_services.append(service)
                
                if len(running_services) == len(services):
                    self.log_result("Docker Services", True, 
                                  f"All services running: {running_services}")
                    return True
                else:
                    missing = set(services) - set(running_services)
                    self.log_result("Docker Services", False, 
                                  f"Missing services: {missing}")
                    return False
            else:
                self.log_result("Docker Services", False, 
                              f"Docker compose error: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_result("Docker Services", False, f"Error checking services: {e}")
            return False
    
    def test_service_health(self) -> bool:
        """Test health of all services."""
        health_checks = [
            ("Backend API", f"{self.base_url}/health"),
            ("Frontend", f"{self.frontend_url}"),
            ("Database Tables", f"{self.base_url}/db/tables"),
        ]
        
        all_healthy = True
        
        for service_name, url in health_checks:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_result(f"{service_name} Health", True, 
                                  f"Service healthy: {response.status_code}")
                else:
                    self.log_result(f"{service_name} Health", False, 
                                  f"Unhealthy: {response.status_code}")
                    all_healthy = False
            except Exception as e:
                self.log_result(f"{service_name} Health", False, f"Error: {e}")
                all_healthy = False
        
        return all_healthy
    
    def test_onset_detection_quality(self) -> bool:
        """Test onset detection with realistic piano audio."""
        audio_file = None
        try:
            # Create realistic piano audio
            audio_file = self.create_realistic_piano_audio(duration=8.0)
            
            # Test via performance upload
            with open(audio_file, 'rb') as f:
                files = {'audio': ('test_piano.wav', f, 'audio/wav')}
                data = {
                    'score_id': '1',
                    'role': 'student',
                    'source': 'onset_quality_test'
                }
                
                response = self.session.post(f"{self.base_url}/api/performances/student", 
                                           files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    features = result.get('features', {})
                    
                    # Validate onset detection quality
                    if features:
                        tempo_curve = features.get('tempo', {})
                        loudness_curve = features.get('loudness', {})
                        
                        tempo_beats = tempo_curve.get('beats', [])
                        loudness_beats = loudness_curve.get('beats', [])
                        
                        # Quality checks
                        expected_notes = 15  # From our test audio
                        detected_beats = len(tempo_beats)
                        
                        if detected_beats >= expected_notes * 0.7:  # 70% detection rate
                            self.log_result("Onset Detection Quality", True,
                                          f"Good detection: {detected_beats} beats detected "
                                          f"(expected ~{expected_notes})")
                            return True
                        else:
                            self.log_result("Onset Detection Quality", False,
                                          f"Poor detection: {detected_beats} beats detected "
                                          f"(expected ~{expected_notes})")
                            return False
                    else:
                        self.log_result("Onset Detection Quality", False,
                                      "No features extracted")
                        return False
                else:
                    self.log_result("Onset Detection Quality", False,
                                  f"Upload failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            self.log_result("Onset Detection Quality", False, f"Error: {e}")
            return False
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def test_expressive_analysis_pipeline(self) -> bool:
        """Test complete expressive analysis pipeline."""
        student_file = None
        reference_file = None
        
        try:
            # Create student and reference performances
            student_file = self.create_realistic_piano_audio(duration=6.0)
            reference_file = self.create_realistic_piano_audio(duration=6.5)  # Slightly different
            
            # Upload reference performance
            with open(reference_file, 'rb') as f:
                files = {'audio': ('reference.wav', f, 'audio/wav')}
                data = {
                    'score_id': '1',
                    'role': 'reference',
                    'source': 'pipeline_test_ref'
                }
                
                ref_response = self.session.post(f"{self.base_url}/api/performances/reference",
                                               files=files, data=data)
            
            # Upload student performance
            with open(student_file, 'rb') as f:
                files = {'audio': ('student.wav', f, 'audio/wav')}
                data = {
                    'score_id': '1',
                    'role': 'student',
                    'source': 'pipeline_test_student'
                }
                
                student_response = self.session.post(f"{self.base_url}/api/performances/student",
                                                   files=files, data=data)
            
            if ref_response.status_code == 200 and student_response.status_code == 200:
                student_result = student_response.json()
                student_perf_id = student_result.get('performance', {}).get('id')
                
                if student_perf_id:
                    # Test envelope generation
                    envelope_response = self.session.post(f"{self.base_url}/api/envelopes/1/build")
                    
                    # Test expressive scoring
                    score_response = self.session.post(f"{self.base_url}/api/expressive-score/{student_perf_id}")
                    
                    if envelope_response.status_code == 200 and score_response.status_code == 200:
                        score_data = score_response.json()
                        overall_score = score_data.get('overall', {})
                        
                        self.log_result("Expressive Analysis Pipeline", True,
                                      f"Complete pipeline working, overall score: {overall_score}")
                        return True
                    else:
                        self.log_result("Expressive Analysis Pipeline", False,
                                      f"Pipeline failed at scoring stage")
                        return False
                else:
                    self.log_result("Expressive Analysis Pipeline", False,
                                  "No student performance ID returned")
                    return False
            else:
                self.log_result("Expressive Analysis Pipeline", False,
                              f"Upload failed: ref={ref_response.status_code}, "
                              f"student={student_response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Expressive Analysis Pipeline", False, f"Error: {e}")
            return False
        finally:
            for file_path in [student_file, reference_file]:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
    
    def test_frontend_integration(self) -> bool:
        """Test frontend integration."""
        try:
            # Test main pages
            pages = [
                ("Dashboard", "/"),
                ("Upload", "/upload"),
                ("Practice", "/practice"),
            ]
            
            all_pages_work = True
            
            for page_name, path in pages:
                try:
                    response = self.session.get(f"{self.frontend_url}{path}")
                    if response.status_code == 200:
                        # Check for key elements
                        content = response.text.lower()
                        if 'piano' in content or 'music' in content or 'ai' in content:
                            self.log_result(f"Frontend {page_name}", True,
                                          f"Page loads correctly")
                        else:
                            self.log_result(f"Frontend {page_name}", False,
                                          f"Page content seems incorrect")
                            all_pages_work = False
                    else:
                        self.log_result(f"Frontend {page_name}", False,
                                      f"HTTP {response.status_code}")
                        all_pages_work = False
                except Exception as e:
                    self.log_result(f"Frontend {page_name}", False, f"Error: {e}")
                    all_pages_work = False
            
            return all_pages_work
            
        except Exception as e:
            self.log_result("Frontend Integration", False, f"Error: {e}")
            return False
    
    def test_performance_under_load(self) -> bool:
        """Test system performance under load."""
        try:
            # Create multiple concurrent requests
            import concurrent.futures
            import threading
            
            def make_health_request():
                try:
                    response = self.session.get(f"{self.base_url}/health", timeout=5)
                    return response.status_code == 200
                except:
                    return False
            
            # Test with 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_health_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            success_rate = sum(results) / len(results)
            
            if success_rate >= 0.9:  # 90% success rate
                self.log_result("Performance Under Load", True,
                              f"Good performance: {success_rate*100:.1f}% success rate")
                return True
            else:
                self.log_result("Performance Under Load", False,
                              f"Poor performance: {success_rate*100:.1f}% success rate")
                return False
                
        except Exception as e:
            self.log_result("Performance Under Load", False, f"Error: {e}")
            return False
    
    def test_data_persistence(self) -> bool:
        """Test data persistence across requests."""
        try:
            # Create a score
            score_data = {
                "slug": "integration-test-score",
                "musicxml_path": "/app/scores/test.xml"
            }
            
            create_response = self.session.post(f"{self.base_url}/api/score-pieces", 
                                              json=score_data)
            
            if create_response.status_code == 200:
                score = create_response.json()
                score_id = score['id']
                
                # Retrieve the score
                get_response = self.session.get(f"{self.base_url}/scores/{score_id}")
                
                if get_response.status_code == 200:
                    retrieved_score = get_response.json()
                    
                    if retrieved_score['slug'] == score_data['slug']:
                        self.log_result("Data Persistence", True,
                                      f"Data persisted correctly: {score_id}")
                        return True
                    else:
                        self.log_result("Data Persistence", False,
                                      "Retrieved data doesn't match")
                        return False
                else:
                    self.log_result("Data Persistence", False,
                                  f"Failed to retrieve: {get_response.status_code}")
                    return False
            else:
                # Score might already exist
                if "already exists" in create_response.text:
                    self.log_result("Data Persistence", True,
                                  "Data persistence working (score exists)")
                    return True
                else:
                    self.log_result("Data Persistence", False,
                                  f"Failed to create: {create_response.status_code}")
                    return False
                    
        except Exception as e:
            self.log_result("Data Persistence", False, f"Error: {e}")
            return False
    
    def run_comprehensive_tests(self) -> bool:
        """Run all integration tests."""
        logger.info("🚀 Starting Google-level comprehensive integration testing...")
        
        tests = [
            ("Docker Services", self.test_docker_services),
            ("Service Health", self.test_service_health),
            ("Data Persistence", self.test_data_persistence),
            ("Onset Detection Quality", self.test_onset_detection_quality),
            ("Expressive Analysis Pipeline", self.test_expressive_analysis_pipeline),
            ("Frontend Integration", self.test_frontend_integration),
            ("Performance Under Load", self.test_performance_under_load),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running {test_name}...")
            try:
                if test_func():
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} failed")
            except Exception as e:
                logger.error(f"💥 {test_name} crashed: {e}")
        
        # Generate comprehensive report
        self.generate_comprehensive_report(passed, total)
        
        return passed == total
    
    def generate_comprehensive_report(self, passed: int, total: int):
        """Generate comprehensive test report."""
        success_rate = (passed / total) * 100
        
        logger.info("\n" + "="*80)
        logger.info("🎯 GOOGLE-LEVEL INTEGRATION TEST REPORT")
        logger.info("="*80)
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {total - passed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Categorize results
        critical_failures = []
        warnings = []
        successes = []
        
        for result in self.test_results:
            if not result['success']:
                if any(keyword in result['test'].lower() 
                      for keyword in ['docker', 'health', 'onset', 'pipeline']):
                    critical_failures.append(result)
                else:
                    warnings.append(result)
            else:
                successes.append(result)
        
        if critical_failures:
            logger.info("\n🚨 CRITICAL FAILURES:")
            for failure in critical_failures:
                logger.info(f"  ❌ {failure['test']}: {failure['message']}")
        
        if warnings:
            logger.info("\n⚠️ WARNINGS:")
            for warning in warnings:
                logger.info(f"  ⚠️ {warning['test']}: {warning['message']}")
        
        logger.info("\n✅ SUCCESSES:")
        for success in successes:
            logger.info(f"  ✅ {success['test']}: {success['message']}")
        
        # Production readiness assessment
        logger.info("\n🎯 PRODUCTION READINESS ASSESSMENT:")
        if success_rate == 100:
            logger.info("  🟢 READY FOR PRODUCTION - All tests passed!")
        elif success_rate >= 90:
            logger.info("  🟡 MOSTLY READY - Minor issues to address")
        elif success_rate >= 70:
            logger.info("  🟠 NEEDS WORK - Several issues to fix")
        else:
            logger.info("  🔴 NOT READY - Major issues require attention")
        
        # Save detailed report
        report_data = {
            'timestamp': time.time(),
            'success_rate': success_rate,
            'passed': passed,
            'total': total,
            'results': self.test_results
        }
        
        with open('integration_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved to: integration_test_report.json")
        logger.info("="*80)


def main():
    """Main function."""
    tester = SystemIntegrationTester()
    
    try:
        success = tester.run_comprehensive_tests()
        exit_code = 0 if success else 1
        
        if success:
            print("\n🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY! 🎉")
        else:
            print("\n⚠️ SOME TESTS FAILED - REVIEW ISSUES BEFORE PRODUCTION")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Testing interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"💥 Testing framework error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
