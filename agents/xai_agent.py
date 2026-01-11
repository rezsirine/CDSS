import numpy as np
from typing import List, Dict, Any, Optional
import warnings

class XAIAgent:
    """
    Agent d'explication et contrefactuels (version simplifiée)
    """
    def __init__(self, model=None, verbose: bool = True):
        """
        Initialise l'agent XAI
        
        Args:
            model: Modèle ML optionnel
            verbose: Mode verbeux
        """
        self.verbose = verbose
        self.model = model
        
        # Essayer d'importer SHAP et LIME (optionnels)
        self.shap_available = False
        self.lime_available = False
        
        try:
            import shap
            self.shap_available = True
            if verbose:
                print("SHAP available")
        except ImportError:
            if verbose:
                print("SHAP not available")
        
        try:
            import lime
            import lime.lime_tabular
            self.lime_available = True
            if verbose:
                print("LIME available")
        except ImportError:
            if verbose:
                print("LIME not available")
        
        if verbose:
            print("XAIAgent initialized")
    
    def generate_explanations(self, 
                            symptoms: List[str], 
                            hypothesis: Dict,
                            feature_values: Optional[List[float]] = None) -> Dict:
        """
        Génère des explications et contrefactuels (version simplifiée)
        
        Args:
            symptoms: Liste des symptômes
            hypothesis: Hypothèse diagnostique
            feature_values: Valeurs des features (optionnel)
            
        Returns:
            Dict avec les explications
        """
        explanations = {
            "symptom_importance": self._compute_symptom_importance(symptoms, hypothesis),
            "counterfactuals": self._generate_counterfactuals(symptoms, hypothesis),
            "sensitivity_analysis": self._sensitivity_analysis(symptoms, hypothesis),
            "feature_contributions": {},
            "method_used": "rule_based"  # Par défaut, méthodes basées sur règles
        }
        
        # Si SHAP est disponible et on a un modèle
        if self.shap_available and self.model is not None and feature_values is not None:
            try:
                shap_values = self._compute_shap_values(feature_values)
                explanations["feature_contributions"]["shap"] = shap_values
                explanations["method_used"] = "shap"
            except Exception as e:
                if self.verbose:
                    print(f"SHAP computation failed: {e}")
        
        # Si LIME est disponible
        if self.lime_available and feature_values is not None:
            try:
                lime_explanation = self._compute_lime_explanation(feature_values, symptoms)
                explanations["feature_contributions"]["lime"] = lime_explanation
                if explanations["method_used"] == "rule_based":
                    explanations["method_used"] = "lime"
            except Exception as e:
                if self.verbose:
                    print(f"LIME computation failed: {e}")
        
        return explanations
    
    def _compute_symptom_importance(self, symptoms: List[str], hypothesis: Dict) -> Dict:
        """Calcule l'importance des symptômes (basé sur règles)"""
        diagnosis = hypothesis.get('diagnosis', '').lower()
        
        # Règles basiques d'importance
        importance_rules = {
            "pneumonia": {"fever": 0.9, "cough": 0.8, "shortness_of_breath": 0.9, "chest_pain": 0.7},
            "influenza": {"fever": 0.8, "cough": 0.7, "fatigue": 0.6, "headache": 0.5},
            "covid-19": {"fever": 0.8, "cough": 0.8, "fatigue": 0.7, "loss_of_taste": 0.9},
            "migraine": {"headache": 0.9, "nausea": 0.7, "sensitivity_to_light": 0.8},
            "gastroenteritis": {"nausea": 0.8, "vomiting": 0.7, "diarrhea": 0.8, "abdominal_pain": 0.9}
        }
        
        # Si le diagnostic est dans nos règles
        if diagnosis in importance_rules:
            rule_importance = importance_rules[diagnosis]
            symptom_importance = {}
            
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                # Chercher une correspondance partielle
                for key, value in rule_importance.items():
                    if key in symptom_lower or symptom_lower in key:
                        symptom_importance[symptom] = value
                        break
                else:
                    # Importance par défaut si non trouvé
                    symptom_importance[symptom] = 0.5
            
            return symptom_importance
        
        # Sinon, calcul basé sur la fréquence
        return {symptom: 0.6 for symptom in symptoms}
    
    def _generate_counterfactuals(self, symptoms: List[str], hypothesis: Dict) -> List[Dict]:
        """Génère des scénarios contrefactuels"""
        diagnosis = hypothesis.get('diagnosis', '')
        
        counterfactuals = []
        
        # 1. Scénario : Retirer le symptôme principal
        if symptoms:
            main_symptom = symptoms[0]
            counterfactuals.append({
                "scenario": f"If {main_symptom} was absent",
                "alternative_diagnoses": ["Viral syndrome", "Stress-related condition"],
                "confidence_change": -0.3,
                "reasoning": f"{main_symptom} is a key indicator for {diagnosis}"
            })
        
        # 2. Scénario : Ajouter un symptôme contradictoire
        contradictory_symptoms = {
            "fever": ["hypothermia"],
            "diarrhea": ["constipation"],
            "headache": ["no_headache"]
        }
        
        for symptom in symptoms[:2]:  # Prendre les 2 premiers symptômes
            if symptom in contradictory_symptoms:
                for contr_symptom in contradictory_symptoms[symptom]:
                    counterfactuals.append({
                        "scenario": f"If {contr_symptom} was present instead of {symptom}",
                        "alternative_diagnoses": ["Different etiology", "Coexisting condition"],
                        "confidence_change": -0.4,
                        "reasoning": f"{symptom} and {contr_symptom} are contradictory"
                    })
        
        # 3. Scénario : Changer la durée
        counterfactuals.append({
            "scenario": "If symptoms lasted only 1 day instead of several days",
            "alternative_diagnoses": ["Acute viral infection", "Transient condition"],
            "confidence_change": -0.2,
            "reasoning": "Duration affects diagnostic certainty"
        })
        
        return counterfactuals[:3]  # Limiter à 3 scénarios
    
    def _sensitivity_analysis(self, symptoms: List[str], hypothesis: Dict) -> Dict:
        """Analyse de sensibilité (quel symptôme change le plus le diagnostic)"""
        sensitivity_scores = {}
        
        for i, symptom in enumerate(symptoms):
            # Score basé sur la position et l'importance perçue
            base_score = 0.7 - (i * 0.1)  # Les premiers symptômes sont plus importants
            sensitivity_scores[symptom] = max(0.3, base_score)
        
        # Trier par sensibilité
        sorted_sensitivity = dict(sorted(sensitivity_scores.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        return {
            "most_sensitive": list(sorted_sensitivity.keys())[:3] if sorted_sensitivity else [],
            "sensitivity_scores": sorted_sensitivity,
            "interpretation": "Symptoms at the beginning of the list have higher impact on diagnosis"
        }
    
    def _compute_shap_values(self, feature_values: List[float]) -> Dict:
        """Calcule les valeurs SHAP (simulé pour l'instant)"""
        # Simulation - à remplacer par vrai calcul SHAP
        n_features = len(feature_values)
        shap_values = {
            "values": [0.1 * (i+1) for i in range(n_features)],
            "base_value": 0.5,
            "data": feature_values,
            "feature_names": [f"feature_{i}" for i in range(n_features)]
        }
        
        return shap_values
    
    def _compute_lime_explanation(self, feature_values: List[float], 
                                feature_names: List[str]) -> Dict:
        """Calcule l'explication LIME (simulé)"""
        # Simulation - à remplacer par vrai LIME
        return {
            "explanation": "LIME explanation would show feature contributions",
            "local_prediction": 0.7,
            "intercept": 0.5,
            "feature_weights": [
                {"feature": name, "weight": 0.1 * (i+1)}
                for i, name in enumerate(feature_names[:5])
            ]
        }
    
    def explain_hypothesis(self, symptoms: List[str], hypothesis: Dict) -> Dict:
        """
        Interface simplifiée pour expliquer une hypothèse
        """
        # Créer des valeurs de features simulées
        feature_values = [1.0 if s in symptoms else 0.0 for s in self._get_all_symptoms()]
        
        explanations = self.generate_explanations(symptoms, hypothesis, feature_values)
        
        return {
            "hypothesis": hypothesis.get('diagnosis', 'Unknown'),
            "confidence": hypothesis.get('confidence', 0.0),
            "explanations": explanations,
            "key_findings": self._extract_key_findings(explanations)
        }
    
    def _get_all_symptoms(self) -> List[str]:
        """Retourne une liste de tous les symptômes connus"""
        return [
            "fever", "cough", "headache", "nausea", "fatigue",
            "shortness_of_breath", "chest_pain", "abdominal_pain",
            "vomiting", "diarrhea", "sore_throat", "runny_nose"
        ]
    
    def _extract_key_findings(self, explanations: Dict) -> List[str]:
        """Extrait les principaux résultats des explications"""
        findings = []
        
        # Importance des symptômes
        symptom_importance = explanations.get("symptom_importance", {})
        if symptom_importance:
            top_symptoms = sorted(symptom_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:3]
            if top_symptoms:
                findings.append(f"Key symptoms: {', '.join([s[0] for s in top_symptoms])}")
        
        # Contrefactuels
        counterfactuals = explanations.get("counterfactuals", [])
        if counterfactuals:
            findings.append(f"{len(counterfactuals)} alternative scenarios considered")
        
        # Sensibilité
        sensitivity = explanations.get("sensitivity_analysis", {})
        if sensitivity.get("most_sensitive"):
            findings.append(f"Most sensitive symptom: {sensitivity['most_sensitive'][0]}")
        
        return findings