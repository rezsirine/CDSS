from typing import List, Dict
from data.ontologies import MedicalOntology

class ValidatorAgent:
    """
    Agent de validation clinique selon ontologies médicales
    """
    def __init__(self):
        self.ontology = MedicalOntology()
        self.rules = self._load_clinical_rules()
        
    def validate_hypothesis(self, 
                           symptoms: List[str], 
                           hypothesis: Dict) -> Dict:
        """
        Valide une hypothèse selon règles cliniques
        """
        validation_results = {
            "ontology_consistency": self._check_ontology(hypothesis, symptoms),
            "temporal_consistency": self._check_temporal_patterns(symptoms),
            "comorbidity_consistency": self._check_comorbidities(hypothesis),
            "severity_alignment": self._check_severity(hypothesis, symptoms)
        }
        
        # Score global de validation
        overall_score = self._compute_validation_score(validation_results)
        
        return {
            "validation_scores": validation_results,
            "overall_score": overall_score,
            "flags": self._generate_red_flags(validation_results)
        }