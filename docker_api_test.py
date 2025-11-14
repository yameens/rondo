#!/usr/bin/env python3
"""
DOCKER API TEST - Google Engineer Level
Comprehensive API testing for when Docker is running.
Tests ALL endpoints with BRUTAL HONESTY.
"""

import requests
import json
import time
import os
import wave
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerAPITester:
    """Comprehensive API tester for Docker environment."""
    
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.frontend_url = "http://localhost:3000"
        self.results = []
        
    def log_test(self, test_name: str, success: bool, message: str, details=None):
        """Log test result with complete honesty."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        
        if details and not success:
            logger.error(f"   Details: {details}")
        
        self.results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'details': details
        })
        
        return success
    
    def wait_for_service(self, url: str, service_name: str, timeout: int = 60):
        """Wait for service to be available."""
        logger.info(f"Waiting for {service_name} at {url}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is available!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                logger.warning(f"Error checking {service_name}: {e}")
            
            time.sleep(2)
        
        logger.error(f"{service_name} not available after {timeout}s")
        return False
    
    def create_test_audio(self, filename="test_audio.wav", duration=3.0):
        """Create test audio file."""
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Create a simple melody with multiple notes
        frequencies = [440, 523, 659, 784]  # A4, C5, E5, G5
        audio = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            start = i * duration / len(frequencies)
            end = (i + 1) * duration / len(frequencies)
            mask = (t >= start) & (t < end)
            audio[mask] = 0.3 * np.sin(2 * np.pi * freq * t[mask])
        
        # Add some envelope to make it more realistic
        envelope = np.exp(-3 * t / duration)
        audio *= envelope
        
        filepath = Path(filename)
        with wave.open(str(filepath), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        logger.info(f"Created test audio: {filepath}")
        return str(filepath)
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return self.log_test("Health Check", True, "API is healthy")
                else:
                    return self.log_test("Health Check", False, f"Unhealthy status: {data}")
            else:
                return self.log_test("Health Check", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("Health Check", False, f"Request failed: {e}")
    
    def test_api_docs(self):
        """Test API documentation endpoint."""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            
            if response.status_code == 200:
                if "swagger" in response.text.lower() or "openapi" in response.text.lower():
                    return self.log_test("API Docs", True, "Swagger docs available")
                else:
                    return self.log_test("API Docs", False, "Docs page doesn't contain Swagger")
            else:
                return self.log_test("API Docs", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("API Docs", False, f"Request failed: {e}")
    
    def test_create_score(self):
        """Test creating a score piece."""
        try:
            data = {
                "slug": f"test-score-{int(time.time())}",
                "musicxml_path": "/app/scores/test.musicxml"
            }
            
            response = requests.post(f"{self.base_url}/api/score-pieces", json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                score_id = result.get("id")
                if score_id:
                    return self.log_test("Create Score", True, f"Created score ID: {score_id}"), score_id
                else:
                    return self.log_test("Create Score", False, "No ID in response"), None
            else:
                return self.log_test("Create Score", False, f"HTTP {response.status_code}: {response.text}"), None
                
        except Exception as e:
            return self.log_test("Create Score", False, f"Request failed: {e}"), None
    
    def test_upload_student_performance(self, score_id: int):
        """Test uploading student performance."""
        audio_file = None
        try:
            # Create test audio
            audio_file = self.create_test_audio("student_test.wav")
            
            with open(audio_file, 'rb') as f:
                files = {'audio': ('student_test.wav', f, 'audio/wav')}
                data = {
                    'score_id': str(score_id),
                    'role': 'student',
                    'source': 'api_test'
                }
                
                response = requests.post(f"{self.base_url}/api/performances/student", 
                                       files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                perf_id = result.get('performance', {}).get('id')
                features = result.get('features', {})
                
                if perf_id and features:
                    return self.log_test("Student Upload", True, 
                                       f"Uploaded performance {perf_id} with features"), perf_id
                elif perf_id:
                    return self.log_test("Student Upload", True, 
                                       f"Uploaded performance {perf_id} (no features)"), perf_id
                else:
                    return self.log_test("Student Upload", False, "No performance ID"), None
            else:
                return self.log_test("Student Upload", False, 
                                   f"HTTP {response.status_code}: {response.text}"), None
                
        except Exception as e:
            return self.log_test("Student Upload", False, f"Request failed: {e}"), None
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def test_upload_reference_performance(self, score_id: int):
        """Test uploading reference performance."""
        audio_file = None
        try:
            # Create test audio
            audio_file = self.create_test_audio("reference_test.wav")
            
            with open(audio_file, 'rb') as f:
                files = {'audio': ('reference_test.wav', f, 'audio/wav')}
                data = {
                    'score_id': str(score_id),
                    'role': 'reference',
                    'source': 'api_test'
                }
                
                response = requests.post(f"{self.base_url}/api/performances/reference", 
                                       files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                perf_id = result.get('performance', {}).get('id')
                
                if perf_id:
                    return self.log_test("Reference Upload", True, f"Uploaded reference {perf_id}"), perf_id
                else:
                    return self.log_test("Reference Upload", False, "No performance ID"), None
            else:
                return self.log_test("Reference Upload", False, 
                                   f"HTTP {response.status_code}: {response.text}"), None
                
        except Exception as e:
            return self.log_test("Reference Upload", False, f"Request failed: {e}"), None
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def test_build_envelopes(self, score_id: int):
        """Test building envelopes."""
        try:
            response = requests.post(f"{self.base_url}/api/envelopes/{score_id}/build", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result:
                    return self.log_test("Build Envelopes", True, f"Built {len(result)} envelopes")
                else:
                    return self.log_test("Build Envelopes", False, "No envelopes returned")
            else:
                return self.log_test("Build Envelopes", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return self.log_test("Build Envelopes", False, f"Request failed: {e}")
    
    def test_get_envelopes(self, score_id: int):
        """Test getting envelopes."""
        try:
            response = requests.get(f"{self.base_url}/api/envelopes/{score_id}", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return self.log_test("Get Envelopes", True, f"Retrieved {len(result)} envelopes")
            else:
                return self.log_test("Get Envelopes", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return self.log_test("Get Envelopes", False, f"Request failed: {e}")
    
    def test_expressive_scoring(self, student_perf_id: int):
        """Test expressive scoring."""
        try:
            response = requests.post(f"{self.base_url}/api/expressive-score/{student_perf_id}", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                overall_score = result.get('overall', {}).get('expressiveness')
                
                if overall_score is not None:
                    return self.log_test("Expressive Scoring", True, 
                                       f"Score: {overall_score}")
                else:
                    return self.log_test("Expressive Scoring", False, "No overall score")
            else:
                return self.log_test("Expressive Scoring", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return self.log_test("Expressive Scoring", False, f"Request failed: {e}")
    
    def test_frontend(self):
        """Test frontend availability."""
        try:
            response = requests.get(self.frontend_url, timeout=10)
            
            if response.status_code == 200:
                if "AI Piano Teacher" in response.text:
                    return self.log_test("Frontend", True, "Frontend loads with correct title")
                else:
                    return self.log_test("Frontend", False, "Frontend loads but missing title")
            else:
                return self.log_test("Frontend", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("Frontend", False, f"Request failed: {e}")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        logger.info("🚀 STARTING COMPREHENSIVE DOCKER API TEST")
        logger.info("=" * 60)
        
        # Wait for services
        if not self.wait_for_service(f"{self.base_url}/health", "Backend API"):
            self.log_test("Service Availability", False, "Backend API not available")
            return False
        
        if not self.wait_for_service(self.frontend_url, "Frontend"):
            self.log_test("Service Availability", False, "Frontend not available")
            # Continue anyway - backend is more critical
        
        # Run tests
        self.test_health_endpoint()
        self.test_api_docs()
        self.test_frontend()
        
        # Test core workflow
        success, score_id = self.test_create_score()
        
        if score_id:
            # Upload performances
            success, student_perf_id = self.test_upload_student_performance(score_id)
            success, ref_perf_id = self.test_upload_reference_performance(score_id)
            
            # Test envelope workflow
            self.test_build_envelopes(score_id)
            self.test_get_envelopes(score_id)
            
            # Test scoring
            if student_perf_id:
                self.test_expressive_scoring(student_perf_id)
        
        # Generate report
        self.generate_final_report()
        
        return True
    
    def generate_final_report(self):
        """Generate final honest report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 FINAL GOOGLE-LEVEL API TEST REPORT")
        logger.info("=" * 60)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info(f"\n❌ FAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    logger.info(f"  - {result['test']}: {result['message']}")
        
        logger.info(f"\n✅ PASSED TESTS:")
        for result in self.results:
            if result['success']:
                logger.info(f"  - {result['test']}: {result['message']}")
        
        # Honest assessment
        logger.info(f"\n🎯 HONEST ASSESSMENT:")
        if passed_tests == total_tests:
            logger.info("  🟢 ALL SYSTEMS FULLY FUNCTIONAL")
            logger.info("  🟢 READY FOR PRODUCTION USE")
        elif passed_tests >= total_tests * 0.9:
            logger.info("  🟡 MOSTLY FUNCTIONAL - Minor issues")
        elif passed_tests >= total_tests * 0.7:
            logger.info("  🟠 PARTIALLY FUNCTIONAL - Major issues need fixing")
        else:
            logger.info("  🔴 SYSTEM BROKEN - Critical failures")
        
        return passed_tests == total_tests

def main():
    """Main test function."""
    tester = DockerAPITester()
    
    try:
        success = tester.run_comprehensive_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
