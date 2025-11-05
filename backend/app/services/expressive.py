"""
Expressive feature envelope services for aggregating reference performances
and computing student deviations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sqlalchemy.orm import Session

from ..models import ScorePiece, Performance, Envelope, FeatureEnum
from ..api.schemas import EnvelopeOut, FeatureCurve, ExpressiveFeatures
from ..database import SessionLocal

logger = logging.getLogger(__name__)


def aggregate_envelope(
    score_id: int, 
    refs: List[Performance],
    db: Optional[Session] = None
) -> Dict[str, EnvelopeOut]:
    """
    Aggregate multiple reference performances into statistical envelopes.
    
    Args:
        score_id: ID of the score piece
        refs: List of reference Performance objects with features_json
        db: Optional database session
        
    Returns:
        Dictionary mapping feature names to EnvelopeOut objects
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        # Get the score to access beat grid
        score = db.query(ScorePiece).filter(ScorePiece.id == score_id).first()
        if not score:
            raise ValueError(f"Score {score_id} not found")
        
        # Extract beat grid data immediately while in session
        beats_json = score.beats_json
        
        # Use beat grid from score
        if beats_json and isinstance(beats_json, list):
            beat_grid = beats_json
        else:
            # Fallback beat grid
            beat_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            logger.warning(f"Using fallback beat grid for score {score_id}")
        
        logger.info(f"Aggregating {len(refs)} references for score {score_id} with {len(beat_grid)} beats")
        
        # Collect feature curves from all references
        feature_collections = {
            'tempo': [],
            'loudness': [],
            'articulation': [],
            'pedal': [],
            'balance': []
        }
        
        valid_refs = 0
        for ref in refs:
            if not ref.features_json:
                logger.warning(f"Performance {ref.id} has no features_json, skipping")
                continue
            
            try:
                # Parse features from JSON
                features_data = ref.features_json
                if isinstance(features_data, dict):
                    # Convert to ExpressiveFeatures object
                    features = ExpressiveFeatures(**features_data)
                else:
                    logger.warning(f"Invalid features_json format for performance {ref.id}")
                    continue
                
                # Extract each feature curve and align to beat grid
                for feature_name in feature_collections.keys():
                    feature_curve = getattr(features, feature_name, None)
                    if feature_curve and isinstance(feature_curve, dict):
                        # Convert dict to FeatureCurve
                        curve = FeatureCurve(**feature_curve)
                    elif hasattr(feature_curve, 'beats') and hasattr(feature_curve, 'values'):
                        curve = feature_curve
                    else:
                        # No data for this feature
                        continue
                    
                    # Align curve to score's beat grid
                    aligned_values = align_curve_to_beat_grid(
                        curve.beats, curve.values, beat_grid
                    )
                    
                    if aligned_values:
                        feature_collections[feature_name].append(aligned_values)
                
                valid_refs += 1
                
            except Exception as e:
                logger.error(f"Failed to process features for performance {ref.id}: {e}")
                continue
        
        if valid_refs == 0:
            raise ValueError("No valid reference performances found")
        
        logger.info(f"Successfully processed {valid_refs} reference performances")
        
        # Compute statistical envelopes for each feature
        envelopes = {}
        
        for feature_name, curves_list in feature_collections.items():
            if not curves_list:
                logger.warning(f"No data for feature {feature_name}, skipping")
                continue
            
            # Stack curves into matrix (n_refs x n_beats)
            curves_matrix = np.array(curves_list)
            
            # Handle NaN values and compute percentiles
            p20_values = []
            median_values = []
            p80_values = []
            
            for beat_idx in range(len(beat_grid)):
                beat_values = curves_matrix[:, beat_idx]
                
                # Remove NaN values
                valid_values = beat_values[~np.isnan(beat_values)]
                
                if len(valid_values) > 0:
                    p20 = float(np.percentile(valid_values, 20))
                    median = float(np.percentile(valid_values, 50))
                    p80 = float(np.percentile(valid_values, 80))
                else:
                    # No valid data for this beat
                    p20 = median = p80 = 0.0
                
                p20_values.append(p20)
                median_values.append(median)
                p80_values.append(p80)
            
            # Create envelope
            envelope = EnvelopeOut(
                score_id=score_id,
                feature=FeatureEnum(feature_name),
                beats=beat_grid,
                p20=p20_values,
                median=median_values,
                p80=p80_values,
                n_refs=len(curves_list),
                created_at=None  # Will be set by database
            )
            
            envelopes[feature_name] = envelope
            logger.info(f"Created envelope for {feature_name} with {len(curves_list)} references")
        
        return envelopes
        
    finally:
        if close_db:
            db.close()


def distance_to_envelope(
    student: Performance, 
    envelope: EnvelopeOut,
    db: Optional[Session] = None
) -> Dict[str, List[float]]:
    """
    Compute per-beat distance of student performance to envelope.
    
    Args:
        student: Student Performance object with features_json
        envelope: Statistical envelope to compare against
        db: Optional database session
        
    Returns:
        Dictionary with distance metrics per beat
    """
    try:
        if not student.features_json:
            raise ValueError(f"Student performance {student.id} has no features_json")
        
        # Parse student features
        features_data = student.features_json
        if isinstance(features_data, dict):
            features = ExpressiveFeatures(**features_data)
        else:
            raise ValueError("Invalid features_json format")
        
        # Get the specific feature curve
        feature_name = envelope.feature.value
        feature_curve = getattr(features, feature_name, None)
        
        if not feature_curve:
            raise ValueError(f"Student has no {feature_name} feature")
        
        if isinstance(feature_curve, dict):
            curve = FeatureCurve(**feature_curve)
        else:
            curve = feature_curve
        
        # Align student curve to envelope beat grid
        student_values = align_curve_to_beat_grid(
            curve.beats, curve.values, envelope.beats
        )
        
        if not student_values:
            raise ValueError("Failed to align student curve to envelope")
        
        # Compute distance metrics
        distances = []
        z_scores = []
        in_band_flags = []
        
        for i in range(len(envelope.beats)):
            student_val = student_values[i]
            envelope_median = envelope.median[i]
            envelope_p20 = envelope.p20[i]
            envelope_p80 = envelope.p80[i]
            
            # Compute scaled absolute deviation
            envelope_range = max(envelope_p80 - envelope_p20, 1e-6)  # Avoid division by zero
            scaled_deviation = abs(student_val - envelope_median) / envelope_range
            distances.append(float(scaled_deviation))
            
            # Compute z-score approximation
            # Use IQR-based standard deviation estimate: σ ≈ IQR / 1.35
            iqr = envelope_p80 - envelope_p20
            estimated_std = max(iqr / 1.35, 1e-6)
            z_score = (student_val - envelope_median) / estimated_std
            z_scores.append(float(z_score))
            
            # Check if within envelope band
            in_band = envelope_p20 <= student_val <= envelope_p80
            in_band_flags.append(in_band)
        
        return {
            'scaled_deviations': distances,
            'z_scores': z_scores,
            'in_band': in_band_flags,
            'out_of_band_count': sum(1 for flag in in_band_flags if not flag),
            'mean_deviation': float(np.mean(distances)),
            'max_deviation': float(np.max(distances))
        }
        
    except Exception as e:
        logger.error(f"Failed to compute distance to envelope: {e}")
        raise


def persist_envelopes(
    envelopes: Dict[str, EnvelopeOut],
    db: Optional[Session] = None
) -> List[Envelope]:
    """
    Persist envelopes to database (upsert operation).
    
    Args:
        envelopes: Dictionary of feature name -> EnvelopeOut
        db: Optional database session
        
    Returns:
        List of persisted Envelope objects
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        persisted_envelopes = []
        
        for feature_name, envelope_out in envelopes.items():
            # Check if envelope already exists
            existing = db.query(Envelope).filter(
                Envelope.score_id == envelope_out.score_id,
                Envelope.feature == envelope_out.feature
            ).first()
            
            if existing:
                # Update existing envelope
                existing.beats = envelope_out.beats
                existing.p20 = envelope_out.p20
                existing.median = envelope_out.median
                existing.p80 = envelope_out.p80
                existing.n_refs = envelope_out.n_refs
                
                envelope_obj = existing
                logger.info(f"Updated existing envelope for {feature_name}")
            else:
                # Create new envelope
                envelope_obj = Envelope(
                    score_id=envelope_out.score_id,
                    feature=envelope_out.feature,
                    beats=envelope_out.beats,
                    p20=envelope_out.p20,
                    median=envelope_out.median,
                    p80=envelope_out.p80,
                    n_refs=envelope_out.n_refs
                )
                db.add(envelope_obj)
                logger.info(f"Created new envelope for {feature_name}")
            
            persisted_envelopes.append(envelope_obj)
        
        db.commit()
        
        # Refresh objects to get updated timestamps
        for envelope in persisted_envelopes:
            db.refresh(envelope)
        
        logger.info(f"Successfully persisted {len(persisted_envelopes)} envelopes")
        return persisted_envelopes
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist envelopes: {e}")
        raise
    finally:
        if close_db:
            db.close()


def align_curve_to_beat_grid(
    curve_beats: List[float], 
    curve_values: List[float], 
    target_beats: List[float]
) -> List[float]:
    """
    Align a feature curve to a target beat grid using interpolation.
    
    Args:
        curve_beats: Original beat positions [0, 1]
        curve_values: Feature values at original beats
        target_beats: Target beat positions [0, 1]
        
    Returns:
        Interpolated values at target beat positions
    """
    try:
        if not curve_beats or not curve_values or len(curve_beats) != len(curve_values):
            logger.warning("Invalid curve data for alignment")
            return [0.0] * len(target_beats)
        
        # Convert to numpy arrays for interpolation
        curve_beats_np = np.array(curve_beats)
        curve_values_np = np.array(curve_values)
        target_beats_np = np.array(target_beats)
        
        # Handle edge cases
        if len(curve_beats) == 1:
            # Single point: repeat value
            return [curve_values[0]] * len(target_beats)
        
        # Ensure beats are sorted
        sort_idx = np.argsort(curve_beats_np)
        curve_beats_sorted = curve_beats_np[sort_idx]
        curve_values_sorted = curve_values_np[sort_idx]
        
        # Interpolate to target beat grid
        aligned_values = np.interp(
            target_beats_np, 
            curve_beats_sorted, 
            curve_values_sorted
        )
        
        return aligned_values.tolist()
        
    except Exception as e:
        logger.error(f"Failed to align curve to beat grid: {e}")
        return [0.0] * len(target_beats)


def get_envelope_for_score(
    score_id: int, 
    feature: FeatureEnum,
    db: Optional[Session] = None
) -> Optional[EnvelopeOut]:
    """
    Retrieve a specific envelope from the database.
    
    Args:
        score_id: Score piece ID
        feature: Feature type
        db: Optional database session
        
    Returns:
        EnvelopeOut object or None if not found
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        envelope = db.query(Envelope).filter(
            Envelope.score_id == score_id,
            Envelope.feature == feature
        ).first()
        
        if envelope:
            return EnvelopeOut(
                score_id=envelope.score_id,
                feature=envelope.feature,
                beats=envelope.beats,
                p20=envelope.p20,
                median=envelope.median,
                p80=envelope.p80,
                n_refs=envelope.n_refs,
                created_at=envelope.created_at
            )
        
        return None
        
    finally:
        if close_db:
            db.close()


def compute_and_persist_envelopes(
    score_id: int,
    db: Optional[Session] = None
) -> Dict[str, EnvelopeOut]:
    """
    Compute envelopes from all reference performances for a score and persist them.
    
    Args:
        score_id: Score piece ID
        db: Optional database session
        
    Returns:
        Dictionary of computed envelopes
    """
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        # Get all reference performances for this score
        refs = db.query(Performance).filter(
            Performance.score_id == score_id,
            Performance.role == 'reference',
            Performance.features_json.isnot(None)
        ).all()
        
        if len(refs) < 1:
            raise ValueError(f"No reference performances found for score {score_id}")
        
        logger.info(f"Computing envelopes from {len(refs)} reference performances")
        
        # Aggregate envelopes
        envelopes = aggregate_envelope(score_id, refs, db)
        
        # Persist to database
        persist_envelopes(envelopes, db)
        
        return envelopes
        
    finally:
        if close_db:
            db.close()
