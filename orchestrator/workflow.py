"""
orchestrator/workflow_final.py - Version CORRECTE avec run_sync
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import traceback
import sys
from pathlib import Path

@dataclass
class WorkflowState:
    """État du workflow clinique"""
    patient_input: str = ""
    symptoms: List[Dict] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    validations: List[Dict] = field(default_factory=list)
    explanations: Dict = field(default_factory=dict)
    confidence_scores: Dict = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    final_diagnosis: Optional[str] = None
    confidence_level: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class ClinicalWorkflow:
    """
    Orchestration du workflow multi-agents
    """
    
    def __init__(self, verbose: bool = True, use_gpu: bool = False):
        """
        Initialise le workflow clinique
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        
        if verbose:
            print("Initializing Clinical Workflow...")
        
        # Initialiser les agents
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialise tous les agents"""
        try:
            from agents.symptom_agent import SymptomAgent
            self.agents['symptom'] = SymptomAgent(verbose=self.verbose)
        except ImportError:
            self.agents['symptom'] = None
            
        try:
            from agents.hypothesis_agent_rag import HypothesisAgentWithRAG as HypothesisAgent
            self.agents['hypothesis'] = HypothesisAgent(
                use_gpu=self.use_gpu, 
                verbose=self.verbose
            )
        except ImportError:
            self.agents['hypothesis'] = None
            
        try:
            from agents.validator_agent import ValidatorAgent
            self.agents['validator'] = ValidatorAgent(verbose=self.verbose)
        except ImportError:
            self.agents['validator'] = None
            
        try:
            from agents.xai_agent import XAIAgent
            self.agents['xai'] = XAIAgent(verbose=self.verbose)
        except ImportError:
            self.agents['xai'] = None
            
        try:
            from agents.confidence_agent import ConfidenceAgent
            self.agents['confidence'] = ConfidenceAgent(verbose=self.verbose)
        except ImportError:
            self.agents['confidence'] = None
    
    async def run(self, patient_input: str) -> Dict:
        """Exécute le workflow asynchrone"""
        state = WorkflowState(patient_input=patient_input)
        state.start_time = datetime.now()
        
        try:
            # 1. Symptom extraction
            if self.verbose:
                print("Extracting symptoms...")
            if self.agents['symptom']:
                results = self.agents['symptom'].extract_from_text(patient_input)
                state.symptoms = results.get('symptoms', [])
            
            # 2. Hypothesis generation
            if state.symptoms and self.agents['hypothesis']:
                if self.verbose:
                    print("Generating hypotheses...")
                symptom_texts = [s['symptom'] for s in state.symptoms]
                hypotheses = self.agents['hypothesis'].generate_hypotheses(symptom_texts)
                if isinstance(hypotheses, list):
                    state.hypotheses = hypotheses
                elif isinstance(hypotheses, dict) and 'hypotheses' in hypotheses:
                    state.hypotheses = hypotheses['hypotheses']
                else:
                    state.hypotheses = []
            
            # 3. Validation
            if state.hypotheses and self.agents['validator']:
                if self.verbose:
                    print("Validating hypotheses...")
                validated = []
                for hyp in state.hypotheses:
                    if isinstance(hyp, dict):
                        symptom_texts = [s['symptom'] for s in state.symptoms]
                        validation = self.agents['validator'].validate_hypothesis(symptom_texts, hyp)
                        validated.append({**hyp, 'validation': validation})
                state.hypotheses = validated
            
            # 4. Confidence scoring
            if state.hypotheses and self.agents['confidence']:
                if self.verbose:
                    print("Computing confidence...")
                hypothesis_probs = {f'h{i}': h.get('confidence', 0.5) for i, h in enumerate(state.hypotheses)}
                validation_scores = {f'h{i}': h.get('validation', {}).get('score', 0.5) for i, h in enumerate(state.hypotheses)}
                xai_scores = {f'h{i}': 0.6 for i in range(len(state.hypotheses))}
                
                confidence = self.agents['confidence'].compute_confidence_score(
                    hypothesis_probs, validation_scores, xai_scores,
                    symptoms_count=len(state.symptoms)
                )
                state.confidence_scores = confidence
                state.confidence_level = confidence.get('composite_confidence', 0.0)
                
        except Exception as e:
            error_msg = f"Workflow error: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
            state.errors.append(error_msg)
        
        state.end_time = datetime.now()
        state.total_duration = (state.end_time - state.start_time).total_seconds()
        
        if self.verbose:
            print(f"Workflow completed in {state.total_duration:.2f} seconds")
        
        return {
            'patient_input': patient_input,
            'symptoms': state.symptoms,
            'hypotheses': state.hypotheses,
            'confidence_scores': state.confidence_scores,
            'confidence_level': state.confidence_level,
            'metadata': {
                'total_symptoms': len(state.symptoms),
                'total_hypotheses': len(state.hypotheses),
                'processing_time': state.total_duration,
                'errors': state.errors
            }
        }
    
    def run_sync(self, patient_input: str) -> Dict:
        """
        Exécute le workflow de manière synchrone
        """
        try:
            return asyncio.run(self.run(patient_input))
        except RuntimeError:
            # Si déjà dans une boucle d'événements
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.run(patient_input))
            finally:
                loop.close()

# Test function
def test_workflow():
    """Teste le workflow"""
    print("Testing ClinicalWorkflow...")
    workflow = ClinicalWorkflow(verbose=True)
    results = workflow.run_sync("Patient has fever and cough")
    print(f"Symptoms: {len(results['symptoms'])}")
    print(f"Confidence: {results.get('confidence_level', 0):.2f}")
    return results

if __name__ == "__main__":
    test_workflow()

