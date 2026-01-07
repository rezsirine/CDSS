import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve

class ConfidenceAgent:
    """
    Agent de calcul de scores de confiance clinique
    """
    def __init__(self, calibration_method="isotonic"):
        self.calibration_method = calibration_method
        self.calibrator = None
        
    def compute_confidence_score(self, 
                                hypothesis_probs: Dict[str, float],
                                validation_scores: Dict[str, float],
                                xai_scores: Dict[str, float]) -> Dict:
        """
        Calcule un score de confiance composite
        """
        # 1. Incertitude épistémique (modèle)
        epistemic_uncertainty = 1 - np.max(list(hypothesis_probs.values()))
        
        # 2. Score de validation clinique
        clinical_consistency = np.mean(list(validation_scores.values()))
        
        # 3. Score d'explicabilité
        explainability_score = np.mean(list(xai_scores.values()))
        
        # 4. Score composite (poids ajustables)
        composite_score = (
            0.4 * (1 - epistemic_uncertainty) +
            0.4 * clinical_consistency +
            0.2 * explainability_score
        )
        
        return {
            "composite_confidence": float(composite_score),
            "epistemic_uncertainty": float(epistemic_uncertainty),
            "clinical_consistency": float(clinical_consistency),
            "explainability_score": float(explainability_score),
            "calibrated_score": self.calibrate_score(composite_score)
        }
    
    def calibrate_score(self, raw_score: float) -> float:
        """Calibration des scores pour fiabilité clinique"""
        # Implémentation de calibration (Isotonic, Platt scaling)
        if self.calibrator:
            return self.calibrator.predict([[raw_score]])[0]
        return raw_score