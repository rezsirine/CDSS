from typing import List, Dict, Optional

class MedicalOntology:
    """Ontologies médicales (simplifiée pour le développement)"""
    
    def __init__(self):
        # Charger des données de base (simulées pour l'instant)
        self.disease_symptoms = {
            "influenza": ["fever", "cough", "fatigue", "headache", "body_aches"],
            "pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain"],
            "migraine": ["headache", "nausea", "sensitivity_to_light"],
            "gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
            "covid_19": ["fever", "cough", "fatigue", "loss_of_taste"]
        }
        
        self.symptom_relationships = {
            "fever": ["infection", "inflammation"],
            "cough": ["respiratory", "infection"],
            "headache": ["neurological", "vascular"]
        }
    
    def check_consistency(self, diagnosis: str, symptoms: List[str]) -> Dict:
        """Vérifie la cohérence entre diagnostic et symptômes"""
        if diagnosis not in self.disease_symptoms:
            return {
                "consistent": False,
                "score": 0.0,
                "message": f"Diagnosis '{diagnosis}' not found in ontology",
                "matching_symptoms": 0,
                "total_expected": 0
            }
        
        expected_symptoms = self.disease_symptoms[diagnosis]
        matched_symptoms = [s for s in symptoms if s in expected_symptoms]
        
        consistency_score = len(matched_symptoms) / len(expected_symptoms) if expected_symptoms else 0.0
        
        return {
            "consistent": consistency_score >= 0.5,  # Au moins 50% des symptômes attendus
            "score": consistency_score,
            "message": f"Matched {len(matched_symptoms)} out of {len(expected_symptoms)} expected symptoms",
            "matching_symptoms": len(matched_symptoms),
            "total_expected": len(expected_symptoms),
            "matched_symptoms": matched_symptoms,
            "missing_symptoms": [s for s in expected_symptoms if s not in symptoms]
        }
    
    def get_related_diseases(self, symptoms: List[str]) -> List[Dict]:
        """Trouve les maladies liées aux symptômes"""
        related = []
        for disease, disease_symptoms in self.disease_symptoms.items():
            matches = sum(1 for symptom in symptoms if symptom in disease_symptoms)
            if matches > 0:
                coverage = matches / len(disease_symptoms)
                related.append({
                    "disease": disease,
                    "match_count": matches,
                    "coverage": coverage,
                    "expected_symptoms": disease_symptoms
                })
        
        # Trier par nombre de correspondances
        related.sort(key=lambda x: x["match_count"], reverse=True)
        return related
    
    def validate_temporal_pattern(self, symptoms: List[Dict]) -> Dict:
        """Valide les patterns temporels (simplifié)"""
        # Pour l'instant, retourne un résultat factice
        return {
            "valid": True,
            "score": 0.7,
            "patterns_found": ["acute_onset"],
            "recommendations": ["Consider duration of symptoms"]
        }

# Créer une instance globale pour usage facile
ontology = MedicalOntology()