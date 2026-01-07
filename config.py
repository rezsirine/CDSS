"""
Configuration pour le système multi-agents
"""

from pathlib import Path

class Config:
    def __init__(self):
        # Chemins
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        
        # Créer les dossiers
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Modèles
        self.SYMPTOM_NER_MODEL = "samrawal/bert-base-uncased_clinical-ner"
        self.WHISPER_MODEL = "base"
        self.SPACY_MODEL = "en_core_web_sm"
        
        # Paramètres
        self.MAX_HYPOTHESES = 5
        self.CONFIDENCE_THRESHOLD = 0.7
        self.RAG_TOP_K = 10

# Instance globale
config = Config()