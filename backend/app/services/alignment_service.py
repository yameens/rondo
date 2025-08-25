import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import mir_eval
from ..config import settings


class AlignmentService:
    def __init__(self, onset_tolerance_ms: int = None, pitch_tolerance: int = None):
        self.onset_tolerance_ms = onset_tolerance_ms or settings.onset_tolerance_ms
        self.pitch_tolerance = pitch_tolerance or settings.pitch_tolerance
        self.onset_tolerance_s = self.onset_tolerance_ms / 1000.0
    
    def align_performance_to_score(self, score_events: List[Dict], performance_events: List[Dict]) -> Dict[str, Any]:
        """
        Align performance events to score events and classify accuracy.
        
        Args:
            score_events: List of score note events
            performance_events: List of performance note events
            
        Returns:
            Dictionary containing alignment results and accuracy metrics
        """
        # Step 1: Create alignment matrix
        alignment_matrix = self._create_alignment_matrix(score_events, performance_events)
        
        # Step 2: Find optimal matching using Hungarian algorithm
        score_indices, perf_indices = linear_sum_assignment(alignment_matrix)
        
        # Step 3: Classify events
        alignment_results = self._classify_events(score_events, performance_events, score_indices, perf_indices)
        
        # Step 4: Calculate metrics
        metrics = self._calculate_metrics(alignment_results)
        
        # Step 5: Group by measures
        measure_analysis = self._analyze_by_measure(alignment_results)
        
        return {
            "alignment_results": alignment_results,
            "metrics": metrics,
            "measure_analysis": measure_analysis,
            "alignment_matrix": alignment_matrix.tolist()
        }
    
    def _create_alignment_matrix(self, score_events: List[Dict], performance_events: List[Dict]) -> np.ndarray:
        """Create cost matrix for alignment."""
        n_score = len(score_events)
        n_perf = len(performance_events)
        
        # Initialize matrix with high costs
        matrix = np.full((n_score, n_perf), 1000.0)
        
        for i, score_event in enumerate(score_events):
            for j, perf_event in enumerate(performance_events):
                # Calculate alignment cost
                cost = self._calculate_alignment_cost(score_event, perf_event)
                matrix[i, j] = cost
        
        return matrix
    
    def _calculate_alignment_cost(self, score_event: Dict, perf_event: Dict) -> float:
        """Calculate cost between a score event and performance event."""
        # Pitch mismatch penalty
        pitch_diff = abs(score_event["pitch"] - perf_event["pitch"])
        if pitch_diff > self.pitch_tolerance:
            return 1000.0  # High penalty for pitch mismatch
        
        # Onset timing cost
        onset_diff = abs(score_event["onset_s"] - perf_event["onset_s"])
        if onset_diff > self.onset_tolerance_s:
            return 1000.0  # High penalty for timing mismatch
        
        # Combined cost (lower is better)
        pitch_cost = pitch_diff * 10.0  # Weight pitch differences
        timing_cost = onset_diff * 100.0  # Weight timing differences more heavily
        
        return pitch_cost + timing_cost
    
    def _classify_events(self, score_events: List[Dict], performance_events: List[Dict], 
                        score_indices: np.ndarray, perf_indices: np.ndarray) -> List[Dict]:
        """Classify each event based on alignment results."""
        results = []
        
        # Track which events have been matched
        score_matched = set(score_indices)
        perf_matched = set(perf_indices)
        
        # Process matched events
        for score_idx, perf_idx in zip(score_indices, perf_indices):
            score_event = score_events[score_idx]
            perf_event = performance_events[perf_idx]
            
            # Check if this is a valid match (cost not too high)
            cost = self._calculate_alignment_cost(score_event, perf_event)
            
            if cost < 1000.0:  # Valid match
                # Determine accuracy type
                accuracy_type = self._determine_accuracy_type(score_event, perf_event)
                
                results.append({
                    "score_event": score_event,
                    "performance_event": perf_event,
                    "type": "matched",
                    "accuracy_type": accuracy_type,
                    "onset_delta": perf_event["onset_s"] - score_event["onset_s"],
                    "pitch_delta": perf_event["pitch"] - score_event["pitch"],
                    "velocity_delta": perf_event.get("velocity", 80) - score_event.get("velocity", 80),
                    "cost": cost
                })
            else:
                # Invalid match - treat as separate missed/extra events
                results.append({
                    "score_event": score_event,
                    "performance_event": None,
                    "type": "missed",
                    "accuracy_type": "missed",
                    "cost": cost
                })
                
                results.append({
                    "score_event": None,
                    "performance_event": perf_event,
                    "type": "extra",
                    "accuracy_type": "extra",
                    "cost": cost
                })
        
        # Add unmatched score events as missed
        for i, score_event in enumerate(score_events):
            if i not in score_matched:
                results.append({
                    "score_event": score_event,
                    "performance_event": None,
                    "type": "missed",
                    "accuracy_type": "missed",
                    "cost": 1000.0
                })
        
        # Add unmatched performance events as extra
        for i, perf_event in enumerate(performance_events):
            if i not in perf_matched:
                results.append({
                    "score_event": None,
                    "performance_event": perf_event,
                    "type": "extra",
                    "accuracy_type": "extra",
                    "cost": 1000.0
                })
        
        return results
    
    def _determine_accuracy_type(self, score_event: Dict, perf_event: Dict) -> str:
        """Determine the specific type of accuracy for a matched event."""
        onset_diff = abs(perf_event["onset_s"] - score_event["onset_s"])
        pitch_diff = abs(perf_event["pitch"] - score_event["pitch"])
        
        if onset_diff <= self.onset_tolerance_s and pitch_diff <= self.pitch_tolerance:
            return "correct"
        elif onset_diff > self.onset_tolerance_s:
            return "timing_error"
        elif pitch_diff > self.pitch_tolerance:
            return "pitch_error"
        else:
            return "other_error"
    
    def _calculate_metrics(self, alignment_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall accuracy metrics."""
        total_score_notes = sum(1 for r in alignment_results if r["score_event"] is not None)
        total_perf_notes = sum(1 for r in alignment_results if r["performance_event"] is not None)
        
        correct_notes = sum(1 for r in alignment_results if r["accuracy_type"] == "correct")
        missed_notes = sum(1 for r in alignment_results if r["accuracy_type"] == "missed")
        extra_notes = sum(1 for r in alignment_results if r["accuracy_type"] == "extra")
        timing_errors = sum(1 for r in alignment_results if r["accuracy_type"] == "timing_error")
        pitch_errors = sum(1 for r in alignment_results if r["accuracy_type"] == "pitch_error")
        
        # Calculate F1 scores using mir_eval style
        precision = correct_notes / (correct_notes + extra_notes) if (correct_notes + extra_notes) > 0 else 0.0
        recall = correct_notes / (correct_notes + missed_notes) if (correct_notes + missed_notes) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate timing accuracy
        timing_deltas = [r["onset_delta"] for r in alignment_results if r["type"] == "matched"]
        timing_rmse = np.sqrt(np.mean(np.array(timing_deltas) ** 2)) if timing_deltas else 0.0
        
        return {
            "total_score_notes": total_score_notes,
            "total_performance_notes": total_perf_notes,
            "correct_notes": correct_notes,
            "missed_notes": missed_notes,
            "extra_notes": extra_notes,
            "timing_errors": timing_errors,
            "pitch_errors": pitch_errors,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "timing_rmse": timing_rmse,
            "overall_accuracy": correct_notes / total_score_notes if total_score_notes > 0 else 0.0
        }
    
    def _analyze_by_measure(self, alignment_results: List[Dict]) -> Dict[int, Dict[str, Any]]:
        """Group analysis results by measure number."""
        measure_analysis = {}
        
        for result in alignment_results:
            if result["score_event"]:
                measure_num = result["score_event"].get("measure", 1)
            elif result["performance_event"]:
                # For extra notes, assign to nearest measure based on timing
                measure_num = int(result["performance_event"]["onset_s"] / 2.0) + 1  # Rough estimate
            else:
                continue
            
            if measure_num not in measure_analysis:
                measure_analysis[measure_num] = {
                    "correct_notes": 0,
                    "missed_notes": 0,
                    "extra_notes": 0,
                    "timing_errors": 0,
                    "pitch_errors": 0,
                    "total_notes": 0,
                    "accuracy": 0.0
                }
            
            measure_data = measure_analysis[measure_num]
            
            if result["accuracy_type"] == "correct":
                measure_data["correct_notes"] += 1
            elif result["accuracy_type"] == "missed":
                measure_data["missed_notes"] += 1
            elif result["accuracy_type"] == "extra":
                measure_data["extra_notes"] += 1
            elif result["accuracy_type"] == "timing_error":
                measure_data["timing_errors"] += 1
            elif result["accuracy_type"] == "pitch_error":
                measure_data["pitch_errors"] += 1
            
            if result["score_event"]:
                measure_data["total_notes"] += 1
        
        # Calculate accuracy per measure
        for measure_num, data in measure_analysis.items():
            if data["total_notes"] > 0:
                data["accuracy"] = data["correct_notes"] / data["total_notes"]
        
        return measure_analysis
