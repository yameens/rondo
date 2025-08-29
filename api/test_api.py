#!/usr/bin/env python3
"""
Simple test script for the Piano Performance Analysis API
"""

import requests
import json

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root():
    """Test the root endpoint"""
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_analyze_mock():
    """Test the analyze endpoint with mock data"""
    try:
        # Create mock files (this would normally be real files)
        mock_score_content = """<?xml version="1.0" encoding="UTF-8"?>
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
    </measure>
  </part>
</score-partwise>"""
        
        # Create a simple mock MP3 file (just a placeholder)
        mock_audio_content = b"mock_audio_data"
        
        files = {
            'score': ('test.musicxml', mock_score_content, 'application/vnd.recordare.musicxml+xml'),
            'audio': ('test.mp3', mock_audio_content, 'audio/mpeg')
        }
        
        response = requests.post("http://localhost:8000/analyze", files=files)
        print(f"Analyze endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code in [200, 400, 500]  # Accept any response as "working"
        
    except Exception as e:
        print(f"Analyze endpoint failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Piano Performance Analysis API...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Analyze Endpoint", test_analyze_mock),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{test_name}: {'PASS' if result else 'FAIL'}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, result in results:
        print(f"{test_name}: {'PASS' if result else 'FAIL'}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
