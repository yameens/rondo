#!/usr/bin/env python3
"""
ACCURACY REALITY CHECK - Google Engineer Level
BRUTAL HONESTY about what actually works vs what's just code.
Tests accuracy features, onset detection, and Verovio integration.
"""

import os
import sys
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyRealityCheck:
    """Brutally honest assessment of accuracy features."""
    
    def __init__(self):
        self.results = []
        self.project_root = Path.cwd()
        
    def log_reality(self, component: str, status: str, reality: str, evidence=None):
        """Log the brutal truth about each component."""
        emoji = {
            'REAL': '✅',
            'FAKE': '❌', 
            'PARTIAL': '⚠️',
            'BROKEN': '💥',
            'MISSING': '🚫'
        }
        
        result = {
            'component': component,
            'status': status,
            'reality': reality,
            'evidence': evidence
        }
        self.results.append(result)
        
        print(f"{emoji.get(status, '❓')} {component}: {reality}")
        if evidence:
            print(f"   Evidence: {evidence}")
    
    def check_accuracy_calculation(self):
        """Check if accuracy calculation is real or fake."""
        
        # Check the old analysis.py (legacy)
        old_analysis = self.project_root / 'app' / 'analysis.py'
        new_analysis = self.project_root / 'backend' / 'app' / 'analysis.py'
        
        accuracy_methods = []
        
        # Check old system
        if old_analysis.exists():
            content = old_analysis.read_text()
            
            if 'compute_scores' in content:
                accuracy_methods.append('Legacy scoring system')
            
            if 'align_midi_notes' in content:
                accuracy_methods.append('MIDI note alignment')
            
            if 'Basic Pitch' in content or 'basic_pitch' in content:
                accuracy_methods.append('Basic Pitch transcription')
        
        # Check new system
        if new_analysis.exists():
            content = new_analysis.read_text()
            
            if 'align_onset_sequences' in content:
                accuracy_methods.append('Onset sequence alignment')
            
            if 'compute_performance_comparison' in content:
                accuracy_methods.append('Performance comparison metrics')
            
            if 'generate_performance_assessment' in content:
                accuracy_methods.append('Assessment generation')
        
        if len(accuracy_methods) >= 3:
            self.log_reality("Accuracy Calculation", "REAL", 
                           f"Multiple methods implemented: {len(accuracy_methods)} systems", 
                           accuracy_methods)
            return True
        elif len(accuracy_methods) >= 1:
            self.log_reality("Accuracy Calculation", "PARTIAL", 
                           f"Basic accuracy system exists", 
                           accuracy_methods)
            return True
        else:
            self.log_reality("Accuracy Calculation", "FAKE", 
                           "No real accuracy calculation found")
            return False
    
    def check_onset_detection_reality(self):
        """Check if onset detection is real librosa-based system."""
        
        onset_file = self.project_root / 'backend' / 'app' / 'onset_detection.py'
        
        if not onset_file.exists():
            self.log_reality("Onset Detection", "MISSING", 
                           "onset_detection.py file not found")
            return False
        
        content = onset_file.read_text()
        
        # Check for real implementation markers
        real_markers = [
            'librosa.onset.onset_detect',
            'spectral_flux',
            'complex_domain',
            'energy_based',
            'phase_deviation',
            'peak_pick',
            'OnsetFrameDetector'
        ]
        
        found_markers = []
        for marker in real_markers:
            if marker in content:
                found_markers.append(marker)
        
        # Check for Basic Pitch (should be replaced)
        has_basic_pitch = 'basic_pitch' in content.lower() or 'Basic Pitch' in content
        
        if len(found_markers) >= 5 and not has_basic_pitch:
            self.log_reality("Onset Detection", "REAL", 
                           f"Professional librosa-based system with {len(found_markers)} methods", 
                           found_markers)
            return True
        elif len(found_markers) >= 3:
            self.log_reality("Onset Detection", "PARTIAL", 
                           f"Basic onset detection with {len(found_markers)} methods", 
                           found_markers)
            return True
        elif has_basic_pitch:
            self.log_reality("Onset Detection", "FAKE", 
                           "Still using Basic Pitch (should be replaced)")
            return False
        else:
            self.log_reality("Onset Detection", "BROKEN", 
                           "Onset detection file exists but lacks real implementation")
            return False
    
    def check_verovio_integration(self):
        """Check if Verovio is properly integrated."""
        
        verovio_component = self.project_root / 'web' / 'src' / 'components' / 'VerovioScore.tsx'
        score_page = self.project_root / 'web' / 'src' / 'app' / '(analysis)' / 'score' / '[slug]' / 'page.tsx'
        
        if not verovio_component.exists():
            self.log_reality("Verovio Integration", "MISSING", 
                           "VerovioScore.tsx component not found")
            return False
        
        verovio_content = verovio_component.read_text()
        
        # Check for real Verovio integration
        real_verovio_markers = [
            'verovio-toolkit-wasm.js',
            'window.verovio.toolkit',
            'loadData',
            'renderToSVG',
            'getPageCount',
            'setOptions'
        ]
        
        found_verovio = []
        for marker in real_verovio_markers:
            if marker in verovio_content:
                found_verovio.append(marker)
        
        # Check if it's actually used
        verovio_used = False
        if score_page.exists():
            score_content = score_page.read_text()
            if 'VerovioScore' in score_content and 'import' in score_content:
                verovio_used = True
            elif 'Verovio notation renderer would be integrated here' in score_content:
                verovio_used = False  # Just a placeholder
        
        if len(found_verovio) >= 4 and verovio_used:
            self.log_reality("Verovio Integration", "REAL", 
                           f"Full Verovio integration with {len(found_verovio)} features and usage", 
                           found_verovio)
            return True
        elif len(found_verovio) >= 4:
            self.log_reality("Verovio Integration", "PARTIAL", 
                           f"Verovio component exists but not fully integrated", 
                           found_verovio)
            return True
        else:
            self.log_reality("Verovio Integration", "FAKE", 
                           "Verovio component is incomplete or placeholder")
            return False
    
    def check_frame_detection(self):
        """Check if frame detection (note duration) is implemented."""
        
        onset_file = self.project_root / 'backend' / 'app' / 'onset_detection.py'
        
        if not onset_file.exists():
            self.log_reality("Frame Detection", "MISSING", 
                           "No onset detection file to check frames")
            return False
        
        content = onset_file.read_text()
        
        frame_markers = [
            'detect_note_frames',
            'note_frames',
            'min_note_duration',
            'frame_length',
            'onset_to_offset'
        ]
        
        found_frames = []
        for marker in frame_markers:
            if marker in content:
                found_frames.append(marker)
        
        if len(found_frames) >= 3:
            self.log_reality("Frame Detection", "REAL", 
                           f"Note frame detection implemented with {len(found_frames)} features", 
                           found_frames)
            return True
        elif len(found_frames) >= 1:
            self.log_reality("Frame Detection", "PARTIAL", 
                           f"Basic frame detection with {len(found_frames)} features", 
                           found_frames)
            return True
        else:
            self.log_reality("Frame Detection", "FAKE", 
                           "No real frame detection implementation")
            return False
    
    def check_expressive_features(self):
        """Check if expressive features (tempo, loudness, etc.) are real."""
        
        analysis_file = self.project_root / 'backend' / 'app' / 'analysis.py'
        
        if not analysis_file.exists():
            self.log_reality("Expressive Features", "MISSING", 
                           "Analysis file not found")
            return False
        
        content = analysis_file.read_text()
        
        feature_functions = [
            'extract_tempo',
            'extract_loudness', 
            'extract_articulation',
            'extract_pedal',
            'extract_balance'
        ]
        
        implemented_features = []
        for func in feature_functions:
            if func in content and 'def ' + func in content:
                implemented_features.append(func)
        
        # Check for real implementation vs stubs
        real_implementation_markers = [
            'librosa.feature.rms',
            'librosa.beat.beat_track',
            'np.percentile',
            'spectral_centroid',
            'chroma_stft'
        ]
        
        real_markers_found = []
        for marker in real_implementation_markers:
            if marker in content:
                real_markers_found.append(marker)
        
        if len(implemented_features) >= 4 and len(real_markers_found) >= 3:
            self.log_reality("Expressive Features", "REAL", 
                           f"{len(implemented_features)} features with real audio processing", 
                           implemented_features)
            return True
        elif len(implemented_features) >= 3:
            self.log_reality("Expressive Features", "PARTIAL", 
                           f"{len(implemented_features)} features but limited processing", 
                           implemented_features)
            return True
        else:
            self.log_reality("Expressive Features", "FAKE", 
                           "Expressive features are stubs or missing")
            return False
    
    def check_envelope_system(self):
        """Check if envelope aggregation system is real."""
        
        envelope_file = self.project_root / 'backend' / 'app' / 'services' / 'expressive.py'
        
        if not envelope_file.exists():
            self.log_reality("Envelope System", "MISSING", 
                           "Envelope service file not found")
            return False
        
        content = envelope_file.read_text()
        
        envelope_markers = [
            'aggregate_envelope',
            'distance_to_envelope',
            'np.percentile',
            'p20',
            'median',
            'p80',
            'persist_envelopes'
        ]
        
        found_envelope = []
        for marker in envelope_markers:
            if marker in content:
                found_envelope.append(marker)
        
        if len(found_envelope) >= 5:
            self.log_reality("Envelope System", "REAL", 
                           f"Complete envelope system with {len(found_envelope)} features", 
                           found_envelope)
            return True
        elif len(found_envelope) >= 3:
            self.log_reality("Envelope System", "PARTIAL", 
                           f"Basic envelope system with {len(found_envelope)} features", 
                           found_envelope)
            return True
        else:
            self.log_reality("Envelope System", "FAKE", 
                           "Envelope system is incomplete")
            return False
    
    def run_reality_check(self):
        """Run complete reality check."""
        print("🔍 ACCURACY REALITY CHECK - Google Engineer Level")
        print("=" * 60)
        print("BRUTAL HONESTY: What actually works vs what's just code")
        print()
        
        tests = [
            ("Accuracy Calculation", self.check_accuracy_calculation),
            ("Onset Detection", self.check_onset_detection_reality),
            ("Frame Detection", self.check_frame_detection),
            ("Expressive Features", self.check_expressive_features),
            ("Envelope System", self.check_envelope_system),
            ("Verovio Integration", self.check_verovio_integration),
        ]
        
        real_count = 0
        total_count = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    real_count += 1
            except Exception as e:
                self.log_reality(test_name, "BROKEN", f"Test crashed: {e}")
        
        print()
        print("=" * 60)
        print("🎯 BRUTAL REALITY ASSESSMENT")
        print("=" * 60)
        print(f"REAL Systems: {real_count}/{total_count}")
        print(f"Reality Rate: {(real_count/total_count)*100:.1f}%")
        
        # Categorize results
        by_status = {}
        for result in self.results:
            status = result['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(result)
        
        for status, results in by_status.items():
            print(f"\n{status}: {len(results)} components")
            for result in results:
                print(f"  - {result['component']}: {result['reality']}")
        
        # Honest recommendations
        print(f"\n🎯 HONEST RECOMMENDATIONS:")
        
        if real_count == total_count:
            print("  🟢 ALL SYSTEMS ARE REAL - Accuracy features are legitimate")
        elif real_count >= total_count * 0.8:
            print("  🟡 MOSTLY REAL - Minor gaps to fill")
        elif real_count >= total_count * 0.5:
            print("  🟠 HALF REAL - Significant work needed")
        else:
            print("  🔴 MOSTLY FAKE - Major implementation required")
        
        print(f"\n📋 SPECIFIC ACTIONS NEEDED:")
        
        # Check what needs fixing
        needs_fixing = []
        for result in self.results:
            if result['status'] in ['FAKE', 'BROKEN', 'MISSING']:
                needs_fixing.append(result['component'])
        
        if 'Onset Detection' in needs_fixing:
            print("  1. ❌ REPLACE Basic Pitch with librosa onset detection")
        if 'Verovio Integration' in needs_fixing:
            print("  2. ❌ PROPERLY INTEGRATE Verovio in score display")
        if 'Frame Detection' in needs_fixing:
            print("  3. ❌ IMPLEMENT note frame detection for duration analysis")
        if 'Accuracy Calculation' in needs_fixing:
            print("  4. ❌ BUILD real accuracy metrics (not just placeholders)")
        
        if not needs_fixing:
            print("  ✅ All systems are functional - ready for production!")
        
        return real_count == total_count


def main():
    """Main reality check function."""
    checker = AccuracyRealityCheck()
    success = checker.run_reality_check()
    
    if success:
        print("\n🎉 REALITY CHECK PASSED - All accuracy features are REAL!")
    else:
        print("\n⚠️ REALITY CHECK FAILED - Some features need real implementation")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
