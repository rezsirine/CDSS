"""
Symptom Agent - Version simplifiée sans Whisper
"""

from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import re

@dataclass
class Symptom:
    """Classe pour représenter un symptôme extrait"""
    text: str
    type: str
    confidence: float
    source: str
    context: Optional[str] = None
    
    def to_dict(self):
        return {
            "symptom": self.text,
            "type": self.type,
            "confidence": self.confidence,
            "source": self.source,
            "context": self.context
        }

class SymptomAgent:
    """
    Agent d'extraction de symptômes (sans Whisper)
    """
    
    def __init__(self, use_gpu: bool = False, verbose: bool = True):
        self.use_gpu = use_gpu
        self.verbose = verbose

        if verbose:
            print(" Initializing Symptom Agent...")

        # spaCy disabled to avoid torch dependency issues
        self.nlp = None

        # Charger NER clinique (désactivé pour éviter les problèmes de chargement)
        self.ner_pipeline = None

        # Dictionnaire de symptômes
        self.symptom_patterns = {
            'fever': [r'fever', r'temperature', r'febrile', r'fièvre'],
            'headache': [r'headache', r'migraine', r'maux de tête'],
            'nausea': [r'nausea', r'nauseous', r'nausée'],
            'pain': [r'pain', r'ache', r'douleur'],
            'fatigue': [r'fatigue', r'tired', r'fatigué'],
            'cough': [r'cough', r'coughing', r'toux'],
            'shortness of breath': [r'shortness of breath', r'difficulty breathing', r'essoufflement'],
            'chest pain': [r'chest pain', r'douleur thoracique'],
            'dizziness': [r'dizziness', r'vertigo', r'vertiges'],
            'vomiting': [r'vomiting', r'vomit', r'vomissement'],
            'diarrhea': [r'diarrhea', r'diarrhoea', r'diarrhée'],
            'rash': [r'rash', r'skin rash', r'éruption cutanée'],
            'chills': [r'chills', r'shivering', r'frissons']
        }

        if verbose:
            print(" Symptom Agent initialized")
    
    def extract_from_text(self, text: str, include_context: bool = True) -> Dict:
        """Extraire les symptômes d'un texte"""
        results = {
            "text": text,
            "symptoms": [],
            "metadata": {
                "text_length": len(text),
                "extraction_methods_used": []
            }
        }
        
        symptoms = []
        
        # Méthode 1: NER clinique (si disponible)
        if self.ner_pipeline:
            try:
                ner_symptoms = self._extract_with_ner(text)
                symptoms.extend(ner_symptoms)
                results["metadata"]["extraction_methods_used"].append("clinical_ner")
            except Exception as e:
                if self.verbose:
                    print(f" NER extraction failed: {e}")
        
        # Méthode 2: Mots-clés
        keyword_symptoms = self._extract_with_keywords(text, include_context)
        symptoms.extend(keyword_symptoms)
        if keyword_symptoms:
            results["metadata"]["extraction_methods_used"].append("keyword_match")
        
        # Méthode 3: spaCy
        if self.nlp:
            try:
                spacy_symptoms = self._extract_with_spacy(text)
                symptoms.extend(spacy_symptoms)
                results["metadata"]["extraction_methods_used"].append("spacy")
            except Exception as e:
                if self.verbose:
                    print(f" spaCy extraction failed: {e}")
        
        # Dédupliquer
        unique_symptoms = []
        seen = set()
        for symptom in symptoms:
            symptom_text = symptom.text.lower()
            if symptom_text not in seen:
                unique_symptoms.append(symptom)
                seen.add(symptom_text)
        
        # Trier par confiance
        unique_symptoms.sort(key=lambda x: x.confidence, reverse=True)
        
        results["symptoms"] = [s.to_dict() for s in unique_symptoms]
        results["metadata"]["total_symptoms"] = len(unique_symptoms)
        
        return results
    
    def _extract_with_ner(self, text: str) -> List[Symptom]:
        """Extraire avec NER clinique"""
        if not self.ner_pipeline or not text.strip():
            return []
        
        try:
            entities = self.ner_pipeline(text)
            symptoms = []
            
            for entity in entities:
                # Vérifier la structure de l'entité
                if isinstance(entity, dict):
                    entity_group = entity.get("entity_group", "")
                    if entity_group and entity_group in ["SYMPTOM", "DISEASE", "SIGN"]:
                        symptom = Symptom(
                            text=entity.get("word", ""),
                            type=entity_group,
                            confidence=entity.get("score", 0.7),
                            source="clinical_ner"
                        )
                        symptoms.append(symptom)
            
            return symptoms
        except Exception as e:
            if self.verbose:
                print(f" NER extraction error: {e}")
            return []
    
    def _extract_with_keywords(self, text: str, include_context: bool) -> List[Symptom]:
        """Extraire par mots-clés"""
        if not text.strip():
            return []
        
        text_lower = text.lower()
        symptoms = []
        found = set()
        
        for symptom_name, patterns in self.symptom_patterns.items():
            if symptom_name in found:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    context = ""
                    if include_context:
                        start = max(0, match.start() - 50)
                        end = min(len(text_lower), match.end() + 50)
                        context = text_lower[start:end]
                        if start > 0:
                            context = "..." + context
                        if end < len(text_lower):
                            context = context + "..."
                    
                    symptom = Symptom(
                        text=symptom_name,
                        type="SYMPTOM",
                        confidence=0.8,
                        source="keyword_match",
                        context=context
                    )
                    
                    symptoms.append(symptom)
                    found.add(symptom_name)
                    break
        
        return symptoms
    
    def _extract_with_spacy(self, text: str) -> List[Symptom]:
        """Extraire avec spaCy"""
        if not self.nlp or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            symptoms = []
            
            for ent in doc.ents:
                if ent.label_ in ["DISEASE", "SYMPTOM", "PROBLEM"]:
                    symptom = Symptom(
                        text=ent.text,
                        type=ent.label_,
                        confidence=0.7,
                        source="spacy"
                    )
                    symptoms.append(symptom)
            
            return symptoms
        except Exception as e:
            if self.verbose:
                print(f" spaCy extraction error: {e}")
            return []
    
    def visualize_results(self, results: Dict):
        """Visualiser les résultats"""
        print(" SYMPTOM EXTRACTION RESULTS")
        
        
        print(f"\n Text: {results['text'][:100]}...")
        print(f" Total symptoms found: {results['metadata']['total_symptoms']}")
        
        if results['symptoms']:
            print("\n Detected symptoms:")
            for i, symptom in enumerate(results['symptoms'], 1):
                print(f"\n{i}. {symptom['symptom'].upper()}")
                print(f"   Type: {symptom['type']}")
                print(f"   Source: {symptom['source']}")
                print(f"   Confidence: {symptom['confidence']:.2f}")
                if symptom.get('context'):
                    print(f"   Context: {symptom['context']}")
        else:
            print("\n No symptoms detected")
    
    def save_results(self, results: Dict, filename: str = "results.json"):
        """Sauvegarder les résultats en JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f" Results saved to: {filename}")
        except Exception as e:
            print(f" Error saving results: {e}")

# Fonction de test
def test_symptom_agent():
    """Tester l'agent de symptômes"""
    print(" Testing Symptom Agent")
    
    try:
        agent = SymptomAgent(verbose=True)
        
        # Cas de test
        test_cases = [
            "The patient presents with fever, headache, and nausea for 3 days.",
            "I have been experiencing chest pain and shortness of breath since yesterday.",
            "Symptoms include cough, fatigue, and body aches.",
            "Patient reports abdominal pain and vomiting after eating."
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n Test Case {i}:")
            print(f"Text: {text}")
            
            results = agent.extract_from_text(text, include_context=True)
            agent.visualize_results(results)
            
            # Sauvegarder les résultats
            agent.save_results(results, f"test_case_{i}.json")
            
            if i < len(test_cases):
                input("\nPress Enter to continue...")
        
        print("\n All tests completed successfully!")
        
    except Exception as e:
        print(f"\n Error during testing: {e}")
        import traceback
        traceback.print_exc()

# Exécuter les tests si le fichier est exécuté directement
if __name__ == "__main__":
    test_symptom_agent()