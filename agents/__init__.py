"""
Package pour les agents du système clinique
"""

__all__ = [
    'SymptomAgent',
    'HypothesisAgent', 
    'ValidatorAgent',
    'XAIAgent',
    'ConfidenceAgent'
]

# Ces imports permettront d'importer facilement tous les agents
from .symptom_agent import SymptomAgent
from .hypothesis_agent import HypothesisAgent
from .confidence_agent import ConfidenceAgent

# Agents optionnels - les importer avec gestion d'erreur
try:
    from .validator_agent import ValidatorAgent
except ImportError as e:
    print(f"Note: ValidatorAgent not available: {e}")
    # Créer une classe factice
    class ValidatorAgent:
        def __init__(self, *args, **kwargs):
            print("Dummy ValidatorAgent created")
        def validate_hypotheses(self, hypotheses):
            return [{"validation": {"score": 0.7, "valid": True}} for _ in hypotheses]

try:
    from .xai_agent import XAIAgent
except ImportError:
    print("Note: XAIAgent not available")
    XAIAgent = None