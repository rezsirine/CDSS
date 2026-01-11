"""
Validator Agent - Version corrigée
"""

from typing import List, Dict, Any, Optional
from data.ontologies import MedicalOntology

class ValidatorAgent:
    """
    Agent de validation clinique selon ontologies médicales
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ontology = MedicalOntology()
        self.rules = self._load_clinical_rules()
        
        if verbose:
            print("ValidatorAgent initialized")
    
    def _load_clinical_rules(self) -> Dict:
        """Charge des règles cliniques basiques"""
        return {
            "emergency_indicators": [
                "chest_pain",
                "shortness_of_breath", 
                "severe_headache",
                "loss_of_consciousness"
            ],
            "contradictions": {
                "fever": ["hypothermia"],
                "diarrhea": ["constipation"]
            }
        }
    
    def validate_hypothesis(self, symptoms: List[str], hypothesis: Dict) -> Dict:
        """
        Valide une hypothèse selon règles cliniques
        """
        if not hypothesis or 'diagnosis' not in hypothesis:
            return {
                "valid": False,
                "score": 0.0,
                "message": "Invalid hypothesis format"
            }
        
        diagnosis = hypothesis['diagnosis']
        
        # 1. Vérification de cohérence avec l'ontologie
        ontology_result = self.ontology.check_consistency(diagnosis, symptoms)
        
        # 2. Vérification des indicateurs d'urgence
        emergency_flags = self._check_emergency_indicators(symptoms)
        
        # 3. Vérification des contradictions
        contradictions = self._check_contradictions(symptoms)
        
        # 4. Calcul du score global
        overall_score = self._compute_validation_score({
            "ontology": ontology_result,
            "emergency": emergency_flags,
            "contradictions": contradictions
        })
        
        return {
            "valid": overall_score >= 0.5,
            "score": overall_score,
            "validation_details": {
                "ontology_consistency": ontology_result,
                "emergency_indicators": emergency_flags,
                "contradictions": contradictions
            },
            "flags": self._generate_red_flags(ontology_result, emergency_flags, contradictions)
        }
    
    def _check_emergency_indicators(self, symptoms: List[str]) -> Dict:
        """Vérifie la présence d'indicateurs d'urgence"""
        emergency_symptoms = self.rules["emergency_indicators"]
        found_emergencies = [s for s in symptoms if s in emergency_symptoms]
        
        return {
            "has_emergency": len(found_emergencies) > 0,
            "emergency_symptoms": found_emergencies,
            "count": len(found_emergencies),
            "score": 0.8 if len(found_emergencies) == 0 else 0.5  # Pénalité pour symptômes d'urgence
        }
    
    def _check_contradictions(self, symptoms: List[str]) -> Dict:
        """Vérifie les contradictions entre symptômes"""
        contradictions = self.rules["contradictions"]
        found_contradictions = []
        
        for symptom, contradicting in contradictions.items():
            if symptom in symptoms:
                for contr in contradicting:
                    if contr in symptoms:
                        found_contradictions.append(f"{symptom} vs {contr}")
        
        return {
            "has_contradictions": len(found_contradictions) > 0,
            "contradiction_pairs": found_contradictions,
            "count": len(found_contradictions),
            "score": 0.9 if len(found_contradictions) == 0 else 0.6  # Pénalité pour contradictions
        }
    
    def _compute_validation_score(self, validation_results: Dict) -> float:
        """Calcule un score global de validation"""
        weights = {
            "ontology": 0.6,  # Poids principal pour la cohérence ontologique
            "emergency": 0.3,
            "contradictions": 0.1
        }
        
        scores = {
            "ontology": validation_results["ontology"].get("score", 0.5),
            "emergency": validation_results["emergency"].get("score", 0.8),
            "contradictions": validation_results["contradictions"].get("score", 0.9)
        }
        
        # Calcul pondéré
        overall_score = sum(scores[key] * weights[key] for key in weights)
        
        # Ajustements
        ontology_score = validation_results["ontology"].get("score", 0)
        if ontology_score > 0.7:
            overall_score = min(overall_score + 0.1, 1.0)
        elif ontology_score < 0.3:
            overall_score = max(overall_score - 0.1, 0.0)
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _generate_red_flags(self, ontology_result: Dict, emergency_flags: Dict, contradictions: Dict) -> List[str]:
        """Génère des drapeaux rouges cliniques"""
        flags = []
        
        if emergency_flags.get("has_emergency", False):
            flags.append(f"Emergency symptoms: {emergency_flags.get('emergency_symptoms', [])}")
        
        if contradictions.get("has_contradictions", False):
            flags.append(f"Contradictions: {contradictions.get('contradiction_pairs', [])}")
        
        if ontology_result.get("score", 0) < 0.3:
            flags.append("Low ontology consistency")
        
        return flags
    
    def validate_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """Valide une liste d'hypothèses"""
        validated_hypotheses = []
        
        for hypothesis in hypotheses:
            if isinstance(hypothesis, dict) and 'diagnosis' in hypothesis:
                # Extraire les symptômes si disponibles
                symptoms = hypothesis.get('symptoms', [])
                if not symptoms and 'explanation' in hypothesis:
                    # Essayer d'extraire les symptômes de l'explication
                    symptoms = self._extract_symptoms_from_text(hypothesis['explanation'])
                
                validation_result = self.validate_hypothesis(symptoms, hypothesis)
                
                validated_hypotheses.append({
                    **hypothesis,
                    "validation": validation_result
                })
        
        # Trier par score de validation
        validated_hypotheses.sort(key=lambda x: x.get("validation", {}).get("score", 0), reverse=True)
        return validated_hypotheses
    
    def _extract_symptoms_from_text(self, text: str) -> List[str]:
        """Extrait les symptômes d'un texte (simplifié)"""
        common_symptoms = [
            "fever", "headache", "cough", "nausea", "fatigue", 
            "pain", "vomiting", "diarrhea", "shortness of breath"
        ]
        
        found_symptoms = []
        text_lower = text.lower()
        for symptom in common_symptoms:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms

# Exporter
__all__ = ['ValidatorAgent']
