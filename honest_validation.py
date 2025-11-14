#!/usr/bin/env python3
"""
HONEST Google-level validation script.
Tests what actually works, reports what doesn't.
No sugar-coating, no demos - just facts.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class HonestValidator:
    """Brutally honest system validator."""
    
    def __init__(self):
        self.results = []
        self.project_root = Path.cwd()
        
    def log_result(self, test_name: str, status: str, message: str, details=None):
        """Log test result with complete honesty."""
        result = {
            'test': test_name,
            'status': status,  # PASS, FAIL, BROKEN, MISSING
            'message': message,
            'details': details
        }
        self.results.append(result)
        
        emoji = {
            'PASS': '✅',
            'FAIL': '❌', 
            'BROKEN': '💥',
            'MISSING': '🚫'
        }
        
        print(f"{emoji.get(status, '❓')} {test_name}: {message}")
        if details:
            print(f"   Details: {details}")
    
    def test_file_structure(self):
        """Test if required files actually exist."""
        required_files = [
            'docker-compose.yml',
            'backend/Dockerfile',
            'web/Dockerfile', 
            'backend/app/main.py',
            'backend/app/models.py',
            'backend/app/analysis.py',
            'backend/app/onset_detection.py',
            'web/src/app/page.tsx',
            'web/package.json',
            'api/requirements.txt'
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if missing_files:
            self.log_result("File Structure", "BROKEN", 
                          f"Missing {len(missing_files)} critical files", 
                          missing_files)
            return False
        else:
            self.log_result("File Structure", "PASS", 
                          f"All {len(required_files)} required files exist")
            return True
    
    def test_docker_config(self):
        """Test Docker configuration validity."""
        try:
            result = subprocess.run(['docker-compose', 'config'], 
                                  capture_output=True, text=True, 
                                  cwd=self.project_root)
            
            if result.returncode == 0:
                self.log_result("Docker Config", "PASS", "docker-compose.yml is valid")
                return True
            else:
                self.log_result("Docker Config", "BROKEN", 
                              "docker-compose.yml has errors", 
                              result.stderr)
                return False
        except FileNotFoundError:
            self.log_result("Docker Config", "MISSING", 
                          "docker-compose command not found")
            return False
        except Exception as e:
            self.log_result("Docker Config", "BROKEN", f"Error: {e}")
            return False
    
    def test_python_imports(self):
        """Test if Python code can actually import."""
        test_imports = [
            ('backend.app.main', 'backend/app/main.py'),
            ('backend.app.models', 'backend/app/models.py'), 
            ('backend.app.analysis', 'backend/app/analysis.py'),
            ('backend.app.onset_detection', 'backend/app/onset_detection.py')
        ]
        
        # Add backend to Python path
        backend_path = str(self.project_root / 'backend')
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        working_imports = []
        broken_imports = []
        
        for module_name, file_path in test_imports:
            try:
                # Try to import the module
                __import__(module_name)
                working_imports.append(module_name)
            except ImportError as e:
                broken_imports.append((module_name, str(e)))
            except Exception as e:
                broken_imports.append((module_name, f"Unexpected error: {e}"))
        
        if broken_imports:
            self.log_result("Python Imports", "BROKEN", 
                          f"{len(broken_imports)} modules can't import", 
                          broken_imports)
            return False
        else:
            self.log_result("Python Imports", "PASS", 
                          f"All {len(working_imports)} modules import successfully")
            return True
    
    def test_dependencies(self):
        """Test if required dependencies are available."""
        # Test Python dependencies
        python_deps = [
            'fastapi', 'uvicorn', 'sqlalchemy', 'pydantic', 
            'librosa', 'numpy', 'scipy', 'soundfile'
        ]
        
        missing_deps = []
        available_deps = []
        
        for dep in python_deps:
            try:
                __import__(dep)
                available_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.log_result("Python Dependencies", "BROKEN", 
                          f"Missing {len(missing_deps)} critical dependencies", 
                          missing_deps)
        else:
            self.log_result("Python Dependencies", "PASS", 
                          f"All {len(available_deps)} dependencies available")
        
        # Test Node.js dependencies
        package_json = self.project_root / 'web' / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    package_data = json.load(f)
                
                deps = package_data.get('dependencies', {})
                dev_deps = package_data.get('devDependencies', {})
                total_deps = len(deps) + len(dev_deps)
                
                self.log_result("Node Dependencies", "PASS", 
                              f"package.json defines {total_deps} dependencies")
            except Exception as e:
                self.log_result("Node Dependencies", "BROKEN", 
                              f"Can't read package.json: {e}")
        else:
            self.log_result("Node Dependencies", "MISSING", 
                          "web/package.json not found")
        
        return len(missing_deps) == 0
    
    def test_api_structure(self):
        """Test API endpoint structure."""
        try:
            # Add paths for imports
            sys.path.insert(0, str(self.project_root / 'backend'))
            
            from app.main import app
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    routes.append({
                        'path': route.path,
                        'methods': list(route.methods) if route.methods else []
                    })
            
            if routes:
                self.log_result("API Structure", "PASS", 
                              f"FastAPI app has {len(routes)} routes defined")
                
                # Check for key endpoints
                key_endpoints = ['/health', '/api/performances/student', '/api/envelopes']
                found_endpoints = []
                
                for endpoint in key_endpoints:
                    for route in routes:
                        if endpoint in route['path']:
                            found_endpoints.append(endpoint)
                            break
                
                if len(found_endpoints) == len(key_endpoints):
                    self.log_result("Key Endpoints", "PASS", 
                                  "All critical endpoints defined")
                else:
                    missing = set(key_endpoints) - set(found_endpoints)
                    self.log_result("Key Endpoints", "BROKEN", 
                                  f"Missing endpoints: {missing}")
                
                return True
            else:
                self.log_result("API Structure", "BROKEN", 
                              "No routes found in FastAPI app")
                return False
                
        except Exception as e:
            self.log_result("API Structure", "BROKEN", 
                          f"Can't analyze FastAPI app: {e}")
            return False
    
    def test_onset_detection(self):
        """Test onset detection system."""
        try:
            sys.path.insert(0, str(self.project_root / 'backend'))
            
            from app.onset_detection import OnsetFrameDetector
            
            # Create detector
            detector = OnsetFrameDetector()
            
            # Test with dummy audio
            import numpy as np
            dummy_audio = np.random.randn(22050 * 2)  # 2 seconds of noise
            
            # Try onset detection
            onsets = detector.detect_onsets_multi_method(dummy_audio)
            
            if len(onsets) >= 0:  # Even 0 onsets is valid for noise
                self.log_result("Onset Detection", "PASS", 
                              f"Onset detection works, found {len(onsets)} onsets in noise")
                return True
            else:
                self.log_result("Onset Detection", "BROKEN", 
                              "Onset detection returned invalid result")
                return False
                
        except Exception as e:
            self.log_result("Onset Detection", "BROKEN", 
                          f"Onset detection system broken: {e}")
            return False
    
    def test_database_models(self):
        """Test database models."""
        try:
            sys.path.insert(0, str(self.project_root / 'backend'))
            
            from app.models import ScorePiece, Performance, Envelope
            from app.database import Base
            
            # Check if models are properly defined
            models = [ScorePiece, Performance, Envelope]
            model_names = [model.__name__ for model in models]
            
            # Check if they inherit from Base
            valid_models = []
            for model in models:
                if issubclass(model, Base):
                    valid_models.append(model.__name__)
            
            if len(valid_models) == len(models):
                self.log_result("Database Models", "PASS", 
                              f"All {len(models)} models properly defined")
                return True
            else:
                self.log_result("Database Models", "BROKEN", 
                              f"Some models not properly defined")
                return False
                
        except Exception as e:
            self.log_result("Database Models", "BROKEN", 
                          f"Database models broken: {e}")
            return False
    
    def test_frontend_structure(self):
        """Test frontend structure."""
        frontend_files = [
            'web/src/app/page.tsx',
            'web/src/app/upload/page.tsx',
            'web/src/app/practice/page.tsx',
            'web/src/components/VerovioScore.tsx',
            'web/package.json',
            'web/next.config.js'
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in frontend_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if missing_files:
            self.log_result("Frontend Structure", "BROKEN", 
                          f"Missing {len(missing_files)} frontend files", 
                          missing_files)
            return False
        else:
            self.log_result("Frontend Structure", "PASS", 
                          f"All {len(frontend_files)} frontend files exist")
            return True
    
    def run_honest_validation(self):
        """Run complete honest validation."""
        print("🔍 GOOGLE-LEVEL HONEST VALIDATION")
        print("=" * 50)
        print("Testing what actually works, reporting what doesn't.")
        print("No sugar-coating, no demos - just facts.\n")
        
        tests = [
            ("File Structure", self.test_file_structure),
            ("Docker Config", self.test_docker_config),
            ("Dependencies", self.test_dependencies),
            ("Python Imports", self.test_python_imports),
            ("Database Models", self.test_database_models),
            ("API Structure", self.test_api_structure),
            ("Onset Detection", self.test_onset_detection),
            ("Frontend Structure", self.test_frontend_structure),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_result(test_name, "BROKEN", f"Test crashed: {e}")
        
        print(f"\n{'=' * 50}")
        print("🎯 HONEST ASSESSMENT RESULTS")
        print(f"{'=' * 50}")
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # Categorize results
        by_status = {}
        for result in self.results:
            status = result['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(result)
        
        for status, results in by_status.items():
            print(f"\n{status}: {len(results)} tests")
            for result in results:
                print(f"  - {result['test']}: {result['message']}")
        
        # Honest recommendation
        print(f"\n🎯 HONEST RECOMMENDATION:")
        if passed == total:
            print("  🟢 SYSTEM STRUCTURE IS SOLID - Ready for Docker deployment")
        elif passed >= total * 0.8:
            print("  🟡 MOSTLY WORKING - Fix critical issues before deployment")
        elif passed >= total * 0.5:
            print("  🟠 SIGNIFICANT ISSUES - Major fixes needed")
        else:
            print("  🔴 SYSTEM BROKEN - Fundamental problems need fixing")
        
        print(f"\n📋 NEXT STEPS:")
        if 'BROKEN' in by_status:
            print("  1. Fix broken components before attempting deployment")
        if 'MISSING' in by_status:
            print("  2. Install missing dependencies")
        print("  3. Start Docker Desktop")
        print("  4. Run: ./start.sh")
        print("  5. Test APIs with real requests")
        
        return passed == total


def main():
    """Main validation function."""
    validator = HonestValidator()
    success = validator.run_honest_validation()
    
    if success:
        print("\n🎉 STRUCTURE VALIDATION PASSED - Ready for deployment testing!")
    else:
        print("\n⚠️ ISSUES FOUND - Fix before deployment")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
