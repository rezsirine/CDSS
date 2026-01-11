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
            },
            "age_considerations": {
                "pediatric": ["age < 18"],
                "geriatric": ["age > 65"]
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
                "contradictions": contradictions,
                "temporal_consistency": self._check_temporal_patterns(symptoms),
                "comorbidity_consistency": self._check_comorbidities(hypothesis),
                "severity_alignment": self._check_severity(hypothesis, symptoms)
            },
            "flags": self._generate_red_flags(ontology_result, emergency_flags, contradictions),
            "recommendations": self._generate_recommendations(ontology_result, emergency_flags)
        }
    
    def _check_emergency_indicators(self, symptoms: List[str]) -> Dict:
        """Vérifie la présence d'indicateurs d'urgence"""
        emergency_symptoms = self.rules["emergency_indicators"]
        found_emergencies = [s for s in symptoms if s in emergency_symptoms]
        
        return {
            "has_emergency": len(found_emergencies) > 0,
            "emergency_symptoms": found_emergencies,
            "count": len(found_emergencies),
            "score": min(len(found_emergencies) * 0.3, 1.0)  # Pénalité pour symptômes d'urgence
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
            "score": max(0, 1.0 - len(found_contradictions) * 0.2)  # Pénalité pour contradictions
        }
    
    def _check_temporal_patterns(self, symptoms: List[str]) -> Dict:
        """Vérifie les patterns temporels (simplifié)"""
        return self.ontology.validate_temporal_pattern([
            {"symptom": s, "duration": "unknown"} for s in symptoms
        ])
    
    def _check_comorbidities(self, hypothesis: Dict) -> Dict:
        """Vérifie les comorbidités potentielles (simplifié)"""
        return {
            "common_comorbidities": [],
            "risk_factors": [],
            "score": 0.8
        }
    
    def _check_severity(self, hypothesis: Dict, symptoms: List[str]) -> Dict:
        """Vérifie l'alignement de sévérité"""
        # Logique simplifiée
        severe_symptoms = ["severe_headache", "high_fever", "chest_pain"]
        severe_count = sum(1 for s in symptoms if s in severe_symptoms)
        
        return {
            "severity_score": min(severe_count * 0.3, 1.0),
            "severe_symptoms_count": severe_count,
            "requires_urgent_care": severe_count >= 2
        }
    
# Dans la méthode _compute_validation_score, augmentez les scores de base :
def _compute_validation_score(self, validation_results: Dict) -> float:
    """Calcule un score global de validation amélioré"""
    weights = {
        "ontology": 0.5,  # Augmenté
        "emergency": 0.2,
        "contradictions": 0.15,
        "temporal": 0.1,
        "severity": 0.05
    }
    
    # Scores de base plus élevés
    scores = {
        "ontology": validation_results["ontology"].get("score", 0.6),  # Augmenté de 0.5 à 0.6
        "emergency": validation_results["emergency"].get("score", 0.8),  # Augmenté
        "contradictions": validation_results["contradictions"].get("score", 0.9),  # Augmenté
        "temporal": validation_results.get("temporal", {"score": 0.8})["score"],  # Augmenté
        "severity": validation_results.get("severity", {"score": 0.7})["score"]  # Nouveau
    }
    
    overall_score = sum(scores[key] * weights[key] for key in weights)
    
    # Bonus pour les bonnes correspondances
    ontology_score = validation_results["ontology"].get("score", 0)
    if ontology_score > 0.7:
        overall_score = min(overall_score + 0.1, 1.0)
    
    return min(max(overall_score, 0.0), 1.0)
    def _generate_red_flags(self, *validation_results) -> List[str]:
        """Génère des drapeaux rouges cliniques"""
        flags = []
        
        for result in validation_results:
            if isinstance(result, dict):
                if result.get("has_emergency", False):
                    flags.append(f"Emergency symptoms detected: {result.get('emergency_symptoms', [])}")
                if result.get("has_contradictions", False):
                    flags.append(f"Symptom contradictions: {result.get('contradiction_pairs', [])}")
        
        return flags
    
    def _generate_recommendations(self, ontology_result: Dict, emergency_flags: Dict) -> List[str]:
        """Génère des recommandations cliniques"""
        recommendations = []
        
        # Recommandations basées sur l'ontologie
        if ontology_result.get("score", 0) < 0.5:
            recommendations.append("Low symptom match - consider alternative diagnoses")
        
        missing_symptoms = ontology_result.get("missing_symptoms", [])
        if missing_symptoms:
            recommendations.append(f"Ask about missing symptoms: {missing_symptoms}")
        
        # Recommandations d'urgence
        if emergency_flags.get("has_emergency", False):
            recommendations.append("Emergency symptoms present - consider urgent evaluation")
        
        return recommendations
    
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