import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.calibration import calibration_curve
import warnings

class ConfidenceAgent:
    """
    Agent de calcul de scores de confiance clinique (Mesurer la fiabilité du diagnostic.)
    "Confiance = (Poids1 × Certitude) + (Poids2 × Validation) + (Poids3 × Explications)"
    """
    def __init__(self, calibration_method="isotonic", verbose=True):
        self.calibration_method = calibration_method
        self.calibrator = None
        self.verbose = verbose
        self.calibration_data = []
        
        if verbose:
            print("ConfidenceAgent initialized with enhanced scoring")
    
    def compute_confidence_score(self, 
                                hypothesis_probs: Dict[str, float],
                                validation_scores: Dict[str, float],
                                xai_scores: Dict[str, float],
                                symptoms_count: int = 0) -> Dict:
        """
        Calcule un score de confiance composite amélioré
        
        Args:
            hypothesis_probs: Probabilités des hypothèses
            validation_scores: Scores de validation clinique
            xai_scores: Scores d'explicabilité
            symptoms_count: Nombre de symptômes détectés
        """
        # 1. Incertitude épistémique (basée sur la meilleure hypothèse)
        if hypothesis_probs:
            max_prob = max(hypothesis_probs.values())
            epistemic_uncertainty = 1 - max_prob
        else:
            max_prob = 0.5
            epistemic_uncertainty = 0.5
        
        # 2. Score de validation clinique 
        if validation_scores:
            # Poids plus important pour les validations
            clinical_consistency = np.mean(list(validation_scores.values()))
            
            #  si certaines validations sont bonnes
            good_validations = sum(1 for v in validation_scores.values() if v > 0.6)
            if good_validations > 0:
                clinical_consistency = min(clinical_consistency + 0.1, 1.0)
        else:
            clinical_consistency = 0.3  # Valeur par défaut plus basse
        
        # 3. Score d'explicabilité
        if xai_scores:
            explainability_score = np.mean(list(xai_scores.values()))
        else:
            explainability_score = 0.5
        
        # 4. Facteur de richesse symptomatique
        symptom_richness = self._compute_symptom_richness(symptoms_count)
        
        # 5. Score composite avec poids dynamiques
        if symptoms_count >= 3:
            # Plus de symptômes → plus de poids à la validation
            weights = {"epistemic": 0.3, "clinical": 0.5, "xai": 0.2}
        else:
            # Moins de symptômes → plus d'incertitude
            weights = {"epistemic": 0.4, "clinical": 0.4, "xai": 0.2}
        
        composite_score = (
            weights["epistemic"] * (1 - epistemic_uncertainty) +
            weights["clinical"] * clinical_consistency +
            weights["xai"] * explainability_score +
            0.1 * symptom_richness  # Bonus pour richesse symptomatique
        )
        
        # Normaliser entre 0.2 et 1.0
        composite_score = max(0.2, min(composite_score, 1.0))
        
        result = {
            "composite_confidence": float(composite_score),
            "epistemic_uncertainty": float(epistemic_uncertainty),
            "clinical_consistency": float(clinical_consistency),
            "explainability_score": float(explainability_score),
            "symptom_richness": float(symptom_richness),
            "weights_used": weights,
            "top_hypothesis_prob": float(max_prob),
            "calibrated_score": self.calibrate_score(composite_score)
        }
        
        if self.verbose:
            print(f"  Confidence breakdown:")
            print(f"    Top hypothesis: {max_prob:.2f}")
            print(f"    Clinical consistency: {clinical_consistency:.2f}")
            print(f"    Symptom richness: {symptom_richness:.2f}")
        
        return result
    
    def _compute_symptom_richness(self, symptoms_count: int) -> float:
        """Calcule la richesse symptomatique"""
        if symptoms_count == 0:
            return 0.0
        elif symptoms_count == 1:
            return 0.3
        elif symptoms_count == 2:
            return 0.5
        elif symptoms_count == 3:
            return 0.7
        elif symptoms_count == 4:
            return 0.8
        else:
            return 0.9
    
    def calibrate_score(self, raw_score: float) -> float:
        """Calibration des scores pour fiabilité clinique"""
        # Logique de calibration simple
        if raw_score < 0.3:
            return raw_score * 0.8  # Pénaliser les très bas scores
        elif raw_score > 0.8:
            return raw_score * 1.1  # Renforcer les hauts scores
        else:
            return raw_score
    
    def compute_for_hypothesis(self, 
                              hypothesis: Dict,
                              validation_result: Dict,
                              xai_result: Dict) -> Dict:
        """Calcule la confiance pour une hypothèse spécifique"""
        hypothesis_prob = hypothesis.get('confidence', 0.5)
        validation_score = validation_result.get('score', 0.3)
        xai_score = xai_result.get('explanations', {}).get('confidence', 0.5)
        
        return self.compute_confidence_score(
            hypothesis_probs={"hypothesis": hypothesis_prob},
            validation_scores={"validation": validation_score},
            xai_scores={"xai": xai_score}
        )