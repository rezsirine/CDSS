import shap
import lime
import numpy as np
from typing import List, Dict

class XAIAgent:
    """
    Agent d'explication et contrefactuels
    """
    def __init__(self, model):
        self.model = model
        self.explainer = shap.Explainer(model)
        
    def generate_explanations(self, 
                             symptoms: List[str], 
                             hypothesis: Dict) -> Dict:
        """
        Génère des explications et contrefactuels
        """
        # 1. Explication SHAP
        shap_values = self.explainer.shap_values(symptoms)
        
        # 2. Explication LIME
        lime_explanation = self._lime_explanation(symptoms, hypothesis)
        
        # 3. Contrefactuels
        counterfactuals = self._generate_counterfactuals(symptoms, hypothesis)
        
        return {
            "shap_values": shap_values,
            "lime_explanation": lime_explanation,
            "counterfactuals": counterfactuals,
            "feature_importance": self._compute_feature_importance(shap_values),
            "sensitivity_analysis": self._sensitivity_analysis(symptoms, hypothesis)
        }