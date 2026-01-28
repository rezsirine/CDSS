"""
Symptom Agent -  avec spaCy 
Hypothesis Agent → Génère des diagnostics
"""

import spacy
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
import re

@dataclass
class Symptom:
    """Classe pour représenter un symptôme extrait"""
    text: str
    type: str  # e.g., NEUROLOGICAL, RESPIRATORY, CARDIOVASCULAR, etc.
    confidence: float
    source: str
    context: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "symptom": self.text,
            "type": self.type,
            "confidence": self.confidence,
            "source": self.source,
            "context": self.context
        }

class SymptomAgent:
    """
    Agent d'extraction de symptômes avec spaCy
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        if verbose:
            print(" Initializing Symptom Agent with spaCy...")
        
        try:
            # Charger le modèle spaCy
            self.nlp = spacy.load("en_core_web_sm")
            if verbose:
                print(" spaCy model loaded successfully")
        except Exception as e:
            print(f" Could not load spaCy model: {e}")
            self.nlp = None
        
        # Initialiser les patterns pour fallback
        self.symptom_patterns = self._initialize_symptom_patterns()
        
        if verbose:
            print(" Symptom Agent initialized")
    
    def _initialize_symptom_patterns(self):
        """Patterns pour fallback si spaCy échoue"""
        return {
            "fever": {"type": "CONSTITUTIONAL", "patterns": [r"fever", r"temperature"]},
            "headache": {"type": "NEUROLOGICAL", "patterns": [r"headache", r"migraine"]},
            "chest pain": {"type": "CARDIOVASCULAR", "patterns": [r"chest pain"]},
            "shortness of breath": {"type": "RESPIRATORY", "patterns": [r"shortness of breath"]},
            "cough": {"type": "RESPIRATORY", "patterns": [r"cough"]},
            "nausea": {"type": "GASTROINTESTINAL", "patterns": [r"nausea"]},
            "vomiting": {"type": "GASTROINTESTINAL", "patterns": [r"vomit"]},
            "fatigue": {"type": "CONSTITUTIONAL", "patterns": [r"fatigue"]},
        }
    
    def extract_symptoms(self, text: str) -> List[Symptom]:
        """Extraction principale avec spaCy"""
        symptoms = []
        
        # Méthode 1: spaCy NER
        if self.nlp:
            spacy_symptoms = self._extract_with_spacy(text)
            symptoms.extend(spacy_symptoms)
        
        # Méthode 2: Keyword fallback
        keyword_symptoms = self._extract_with_keywords(text)
        symptoms.extend(keyword_symptoms)
        
        # Dédupliquer
        unique_symptoms = []
        seen = set()
        for symptom in symptoms:
            if symptom.text not in seen:
                unique_symptoms.append(symptom)
                seen.add(symptom.text)
        
        return unique_symptoms
    
    def _extract_with_spacy(self, text: str) -> List[Symptom]:
        """Extraction avec spaCy NER"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            symptoms = []
            
            for ent in doc.ents:
                # Détecter les symptômes basés sur les entités
                if ent.label_ in ["DISEASE", "SYMPTOM"]:
                    symptom_type = self._classify_symptom_type(ent.text)
                    
                    symptom = Symptom(
                        text=ent.text,
                        type=symptom_type,
                        confidence=0.8,  # spaCy a une bonne confiance
                        source="spacy_ner"
                    )
                    symptoms.append(symptom)
            
            return symptoms
        except Exception as e:
            if self.verbose:
                print(f" spaCy extraction error: {e}")
            return []
    
    def _extract_with_keywords(self, text: str) -> List[Symptom]:
        """Extraction par mots-clés"""
        text_lower = text.lower()
        symptoms = []
        
        for symptom_name, data in self.symptom_patterns.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    symptom = Symptom(
                        text=symptom_name.title(),
                        type=data["type"],
                        confidence=0.7,
                        source="keyword_match"
                    )
                    symptoms.append(symptom)
                    break
        
        return symptoms
    
    def _classify_symptom_type(self, symptom_text: str) -> str:
        """Classifie le type de symptôme"""
        text_lower = symptom_text.lower()
        
        if any(word in text_lower for word in ["head", "dizzy", "confusion", "migraine"]):
            return "NEUROLOGICAL"
        elif any(word in text_lower for word in ["chest", "heart", "breath", "palpitation"]):
            return "CARDIOVASCULAR"
        elif any(word in text_lower for word in ["cough", "wheez", "respiratory"]):
            return "RESPIRATORY"
        elif any(word in text_lower for word in ["nausea", "vomit", "diarrhea", "abdominal"]):
            return "GASTROINTESTINAL"
        elif any(word in text_lower for word in ["fever", "fatigue", "chills"]):
            return "CONSTITUTIONAL"
        else:
            return "GENERAL"
    
    def extract_from_text(self, text: str) -> Dict:
        """Format pour Streamlit"""
        symptoms = self.extract_symptoms(text)
        
        return {
            "text": text,
            "symptoms": [s.to_dict() for s in symptoms],
            "metadata": {
                "total_symptoms": len(symptoms),
                "methods_used": list(set(s.source for s in symptoms))
            }
        }

# Test
def test():
    agent = SymptomAgent()
    text = "Patient has fever, headache, and chest pain"
    results = agent.extract_from_text(text)
    
    print("Results:")
    for symptom in results["symptoms"]:
        print(f"  {symptom['symptom']}: {symptom['type']}")

if __name__ == "__main__":
    test()