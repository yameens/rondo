import pytest
import numpy as np
from backend.app.services.alignment_service import AlignmentService


class TestAlignmentService:
    @pytest.fixture
    def alignment_service(self):
        return AlignmentService(onset_tolerance_ms=50, pitch_tolerance=0)
    
    @pytest.fixture
    def sample_score_events(self):
        return [
            {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5, "velocity": 80, "measure": 1},
            {"pitch": 64, "onset_s": 2.0, "offset_s": 2.25, "velocity": 80, "measure": 1},
            {"pitch": 67, "onset_s": 3.0, "offset_s": 3.5, "velocity": 80, "measure": 2},
        ]
    
    @pytest.fixture
    def sample_performance_events(self):
        return [
            {"pitch": 60, "onset_s": 1.05, "offset_s": 1.55, "velocity": 85, "measure": 1},
            {"pitch": 64, "onset_s": 2.1, "offset_s": 2.35, "velocity": 80, "measure": 1},
            {"pitch": 67, "onset_s": 3.0, "offset_s": 3.5, "velocity": 80, "measure": 2},
            {"pitch": 69, "onset_s": 4.0, "offset_s": 4.25, "velocity": 80, "measure": 2},  # Extra note
        ]
    
    def test_calculate_alignment_cost_exact_match(self, alignment_service):
        """Test alignment cost calculation for exact matches."""
        score_event = {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5}
        perf_event = {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5}
        
        cost = alignment_service._calculate_alignment_cost(score_event, perf_event)
        
        assert cost == 0.0  # Perfect match should have zero cost
    
    def test_calculate_alignment_cost_timing_error(self, alignment_service):
        """Test alignment cost calculation for timing errors."""
        score_event = {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5}
        perf_event = {"pitch": 60, "onset_s": 1.1, "offset_s": 1.6}  # 100ms late
        
        cost = alignment_service._calculate_alignment_cost(score_event, perf_event)
        
        # Should be high cost due to timing mismatch
        assert cost > 1000.0
    
    def test_calculate_alignment_cost_pitch_error(self, alignment_service):
        """Test alignment cost calculation for pitch errors."""
        score_event = {"pitch": 60, "onset_s": 1.0, "offset_s": 1.5}
        perf_event = {"pitch": 62, "onset_s": 1.0, "offset_s": 1.5}  # Wrong pitch
        
        cost = alignment_service._calculate_alignment_cost(score_event, perf_event)
        
        # Should be high cost due to pitch mismatch
        assert cost > 1000.0
    
    def test_align_performance_to_score(self, alignment_service, sample_score_events, sample_performance_events):
        """Test full alignment process."""
        result = alignment_service.align_performance_to_score(sample_score_events, sample_performance_events)
        
        assert "alignment_results" in result
        assert "metrics" in result
        assert "measure_analysis" in result
        
        # Check that we have alignment results
        alignment_results = result["alignment_results"]
        assert len(alignment_results) > 0
        
        # Check metrics
        metrics = result["metrics"]
        assert metrics["total_score_notes"] == 3
        assert metrics["total_performance_notes"] == 4
        assert metrics["extra_notes"] == 1  # One extra note in performance
    
    def test_determine_accuracy_type_correct(self, alignment_service):
        """Test accuracy type determination for correct notes."""
        score_event = {"pitch": 60, "onset_s": 1.0}
        perf_event = {"pitch": 60, "onset_s": 1.0}
        
        accuracy_type = alignment_service._determine_accuracy_type(score_event, perf_event)
        
        assert accuracy_type == "correct"
    
    def test_determine_accuracy_type_timing_error(self, alignment_service):
        """Test accuracy type determination for timing errors."""
        score_event = {"pitch": 60, "onset_s": 1.0}
        perf_event = {"pitch": 60, "onset_s": 1.1}  # 100ms late
        
        accuracy_type = alignment_service._determine_accuracy_type(score_event, perf_event)
        
        assert accuracy_type == "timing_error"
    
    def test_determine_accuracy_type_pitch_error(self, alignment_service):
        """Test accuracy type determination for pitch errors."""
        score_event = {"pitch": 60, "onset_s": 1.0}
        perf_event = {"pitch": 62, "onset_s": 1.0}  # Wrong pitch
        
        accuracy_type = alignment_service._determine_accuracy_type(score_event, perf_event)
        
        assert accuracy_type == "pitch_error"
    
    def test_calculate_metrics(self, alignment_service):
        """Test metrics calculation."""
        alignment_results = [
            {"accuracy_type": "correct", "type": "matched"},
            {"accuracy_type": "correct", "type": "matched"},
            {"accuracy_type": "missed", "type": "missed"},
            {"accuracy_type": "extra", "type": "extra"},
            {"accuracy_type": "timing_error", "type": "matched"},
        ]
        
        metrics = alignment_service._calculate_metrics(alignment_results)
        
        assert metrics["correct_notes"] == 2
        assert metrics["missed_notes"] == 1
        assert metrics["extra_notes"] == 1
        assert metrics["timing_errors"] == 1
        assert metrics["overall_accuracy"] == 0.5  # 2 correct out of 4 total score notes
    
    def test_analyze_by_measure(self, alignment_service):
        """Test measure-by-measure analysis."""
        alignment_results = [
            {
                "score_event": {"pitch": 60, "measure": 1},
                "performance_event": {"pitch": 60},
                "accuracy_type": "correct"
            },
            {
                "score_event": {"pitch": 64, "measure": 1},
                "performance_event": None,
                "accuracy_type": "missed"
            },
            {
                "score_event": {"pitch": 67, "measure": 2},
                "performance_event": {"pitch": 67},
                "accuracy_type": "correct"
            },
        ]
        
        measure_analysis = alignment_service._analyze_by_measure(alignment_results)
        
        assert 1 in measure_analysis
        assert 2 in measure_analysis
        
        measure_1 = measure_analysis[1]
        assert measure_1["correct_notes"] == 1
        assert measure_1["missed_notes"] == 1
        assert measure_1["total_notes"] == 2
        assert measure_1["accuracy"] == 0.5
        
        measure_2 = measure_analysis[2]
        assert measure_2["correct_notes"] == 1
        assert measure_2["total_notes"] == 1
        assert measure_2["accuracy"] == 1.0
