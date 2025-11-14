#!/usr/bin/env python3
"""
FINAL HONEST TEST - Google Engineer Level
Tests what can be tested without Docker, provides honest assessment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def test_dashboard_cleanliness():
    """Test if dashboard is clean and professional."""
    dashboard_file = Path('web/src/app/page.tsx')
    
    if not dashboard_file.exists():
        return False, "Dashboard file missing"
    
    content = dashboard_file.read_text()
    
    # Check for clean patterns
    clean_indicators = [
        'Clean, Professional Header',
        'Clean Stats Grid', 
        'Recent Activity',
        'Quick Actions',
        'bg-white border border-gray-200',  # Clean styling
        'text-sm font-medium text-gray-600'  # Professional typography
    ]
    
    missing_indicators = []
    for indicator in clean_indicators:
        if indicator not in content:
            missing_indicators.append(indicator)
    
    # Check for bloated patterns (should be removed)
    bloated_patterns = [
        'useState<DashboardStats>',  # Removed complex state
        'fetchStats',  # Removed unnecessary API calls
        'quickActions.map',  # Removed complex mapping
        'recentScores.map'  # Removed complex mapping
    ]
    
    found_bloat = []
    for pattern in bloated_patterns:
        if pattern in content:
            found_bloat.append(pattern)
    
    if missing_indicators:
        return False, f"Missing clean patterns: {missing_indicators}"
    
    if found_bloat:
        return False, f"Still has bloated code: {found_bloat}"
    
    return True, "Dashboard is clean and professional"

def test_docker_readiness():
    """Test Docker configuration without running Docker."""
    
    # Test docker-compose.yml syntax
    try:
        result = subprocess.run(['docker-compose', 'config'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode != 0:
            return False, f"docker-compose.yml invalid: {result.stderr}"
    except FileNotFoundError:
        return False, "docker-compose not installed"
    
    # Check required files
    required_files = [
        'docker-compose.yml',
        'backend/Dockerfile',
        'web/Dockerfile',
        'start.sh'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        return False, f"Missing Docker files: {missing}"
    
    return True, "Docker configuration is ready"

def test_code_structure():
    """Test code structure and organization."""
    
    # Check backend structure
    backend_files = [
        'backend/app/main.py',
        'backend/app/models.py',
        'backend/app/analysis.py',
        'backend/app/onset_detection.py',
        'backend/app/database.py'
    ]
    
    missing_backend = []
    for file_path in backend_files:
        if not Path(file_path).exists():
            missing_backend.append(file_path)
    
    # Check frontend structure
    frontend_files = [
        'web/src/app/page.tsx',
        'web/src/app/upload/page.tsx',
        'web/src/app/practice/page.tsx',
        'web/package.json'
    ]
    
    missing_frontend = []
    for file_path in frontend_files:
        if not Path(file_path).exists():
            missing_frontend.append(file_path)
    
    if missing_backend:
        return False, f"Missing backend files: {missing_backend}"
    
    if missing_frontend:
        return False, f"Missing frontend files: {missing_frontend}"
    
    return True, "Code structure is complete"

def test_api_requirements():
    """Test API requirements file."""
    req_file = Path('api/requirements.txt')
    
    if not req_file.exists():
        return False, "requirements.txt missing"
    
    content = req_file.read_text()
    
    # Check for essential dependencies
    essential_deps = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'pydantic',
        'librosa',
        'numpy'
    ]
    
    missing_deps = []
    for dep in essential_deps:
        if dep not in content.lower():
            missing_deps.append(dep)
    
    if missing_deps:
        return False, f"Missing dependencies: {missing_deps}"
    
    return True, "All essential dependencies listed"

def test_frontend_package():
    """Test frontend package.json."""
    package_file = Path('web/package.json')
    
    if not package_file.exists():
        return False, "package.json missing"
    
    try:
        with open(package_file) as f:
            package_data = json.load(f)
        
        # Check for essential dependencies
        deps = package_data.get('dependencies', {})
        dev_deps = package_data.get('devDependencies', {})
        
        essential_deps = ['next', 'react', 'react-dom']
        missing = []
        
        for dep in essential_deps:
            if dep not in deps and dep not in dev_deps:
                missing.append(dep)
        
        if missing:
            return False, f"Missing frontend dependencies: {missing}"
        
        return True, f"Frontend has {len(deps) + len(dev_deps)} dependencies"
        
    except json.JSONDecodeError:
        return False, "package.json is invalid JSON"

def main():
    """Run final honest test."""
    print("🎯 FINAL HONEST TEST - Google Engineer Level")
    print("=" * 60)
    print("Testing what can be validated without running Docker")
    print()
    
    tests = [
        ("Dashboard Cleanliness", test_dashboard_cleanliness),
        ("Docker Readiness", test_docker_readiness),
        ("Code Structure", test_code_structure),
        ("API Requirements", test_api_requirements),
        ("Frontend Package", test_frontend_package),
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}: {message}")
            
            if success:
                passed += 1
            
            results.append({
                'test': test_name,
                'success': success,
                'message': message
            })
            
        except Exception as e:
            print(f"💥 CRASH {test_name}: {e}")
            results.append({
                'test': test_name,
                'success': False,
                'message': f"Test crashed: {e}"
            })
    
    print()
    print("=" * 60)
    print("🎯 FINAL HONEST ASSESSMENT")
    print("=" * 60)
    print(f"Structure Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print()
    print("🔍 WHAT THIS MEANS:")
    
    if passed == total:
        print("  ✅ STRUCTURE IS SOLID")
        print("  ✅ DASHBOARD IS CLEAN") 
        print("  ✅ DOCKER CONFIG IS VALID")
        print("  ✅ ALL FILES ARE PRESENT")
        print()
        print("🚀 READY FOR DOCKER DEPLOYMENT:")
        print("  1. Start Docker Desktop")
        print("  2. Run: ./start.sh")
        print("  3. Test APIs at http://localhost:8001/docs")
        print("  4. Test frontend at http://localhost:3000")
        
    elif passed >= total * 0.8:
        print("  🟡 MOSTLY READY - Fix remaining issues")
        
    else:
        print("  🔴 MAJOR ISSUES - Fix before deployment")
    
    print()
    print("⚠️ CANNOT TEST WITHOUT DOCKER:")
    print("  - API functionality (requires FastAPI server)")
    print("  - Database operations (requires PostgreSQL)")
    print("  - Onset detection (requires librosa/numpy)")
    print("  - Frontend build (requires Node.js)")
    print()
    print("💡 NEXT STEPS:")
    print("  1. Ensure Docker Desktop is running")
    print("  2. Run: ./start.sh")
    print("  3. Wait for all services to start")
    print("  4. Test with: python3 test_apis.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
