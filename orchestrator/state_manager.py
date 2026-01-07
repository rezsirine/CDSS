from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ClinicalState:
    """Gestion de l'état clinique durant le workflow"""
    patient_id: str
    symptoms: List[Dict]
    extracted_entities: List[Dict]
    generated_hypotheses: List[Dict]
    validation_results: Dict[str, Any]
    explanations: Dict[str, Any]
    confidence_scores: Dict[str, float]
    timestamp: datetime
    metadata: Dict

@dataclass
class WorkflowState:
    """State management for the clinical workflow"""

    # Input
    patient_input: str = ""

    # Processing results
    symptoms: List[Dict] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    validations: List[Dict] = field(default_factory=list)
    explanations: Dict = field(default_factory=dict)
    confidence_scores: Dict = field(default_factory=dict)

    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

    # Final results
    final_diagnosis: Optional[str] = None
    confidence_level: float = 0.0
    recommendations: List[str] = field(default_factory=list)

class StateManager:
    """Gestionnaire d'état centralisé"""
    def __init__(self):
        self.current_state = None
        self.history = []

    def update_state(self, new_state: ClinicalState):
        self.history.append(self.current_state)
        self.current_state = new_state
