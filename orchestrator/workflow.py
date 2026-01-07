from typing import Dict, List
import asyncio
from dataclasses import dataclass
from datetime import datetime
from orchestrator.state_manager import WorkflowState

class ClinicalWorkflow:
    """
    Orchestration du workflow multi-agents avec LangGraph
    """
    def __init__(self, verbose: bool = True, use_gpu: bool = False):
        self.verbose = verbose
        self.use_gpu = use_gpu

        if verbose:
            print(" Initializing Clinical Workflow...")

        # Initialize agents
        self.agents = {}
        self._initialize_agents()

        if verbose:
            print(" Clinical Workflow initialized")

    def _initialize_agents(self):
        """Initialize all agents"""
        try:
            from agents.symptom_agent import SymptomAgent
            self.agents['symptom'] = SymptomAgent(verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f" Failed to load SymptomAgent: {e}")
            self.agents['symptom'] = None

        try:
            from agents.hypothesis_agent import HypothesisAgent
            self.agents['hypothesis'] = HypothesisAgent(use_gpu=self.use_gpu, verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f" Failed to load HypothesisAgent: {e}")
            self.agents['hypothesis'] = None

        try:
            from agents.validator_agent import ValidatorAgent
            self.agents['validator'] = ValidatorAgent(verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f" Failed to load ValidatorAgent: {e}")
            self.agents['validator'] = None

        try:
            from agents.xai_agent import XAIAgent
            self.agents['xai'] = XAIAgent(verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f" Failed to load XAIAgent: {e}")
            self.agents['xai'] = None

        try:
            from agents.confidence_agent import ConfidenceAgent
            self.agents['confidence'] = ConfidenceAgent(verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f" Failed to load ConfidenceAgent: {e}")
            self.agents['confidence'] = None

    async def run(self, patient_input: str) -> Dict:
        """Execute the complete clinical workflow"""
        state = WorkflowState(patient_input=patient_input)
        state.start_time = datetime.now()

        try:
            # Step 1: Symptom Extraction
            if self.verbose:
                print(" Extracting symptoms...")
            state = await self._process_symptoms(state)

            # Step 2: Hypothesis Generation
            if self.verbose:
                print(" Generating hypotheses...")
            state = await self._process_hypotheses(state)

            # Step 3: Clinical Validation
            if self.verbose:
                print(" Validating clinically...")
            state = await self._process_validation(state)

            # Step 4: XAI Explanations
            if self.verbose:
                print(" Generating explanations...")
            state = await self._process_explanations(state)

            # Step 5: Confidence Scoring
            if self.verbose:
                print(" Computing confidence...")
            state = await self._process_confidence(state)

        except Exception as e:
            error_msg = f"Workflow error: {str(e)}"
            if self.verbose:
                print(f" {error_msg}")
            state.errors.append(error_msg)

        # Finalize
        state.end_time = datetime.now()
        if state.start_time:
            state.total_duration = (state.end_time - state.start_time).total_seconds()

        if self.verbose:
            print(f"Workflow completed in {state.total_duration:.2f} seconds")
        return self._format_results(state)

    def run_sync(self, patient_input: str) -> Dict:
        """Synchronous wrapper for the async run method"""
        return asyncio.run(self.run(patient_input))

    async def _process_symptoms(self, state: WorkflowState) -> WorkflowState:
        """Process symptom extraction"""
        if self.agents['symptom']:
            try:
                results = self.agents['symptom'].extract_from_text(state.patient_input)
                state.symptoms = results.get('symptoms', [])
            except Exception as e:
                state.errors.append(f"Symptom extraction failed: {e}")
        return state

    async def _process_hypotheses(self, state: WorkflowState) -> WorkflowState:
        """Process hypothesis generation"""
        if self.agents['hypothesis'] and state.symptoms:
            try:
                symptom_texts = [s['symptom'] for s in state.symptoms]
                results = self.agents['hypothesis'].generate_hypotheses(symptom_texts)
                state.hypotheses = results.get('hypotheses', [])
            except Exception as e:
                state.errors.append(f"Hypothesis generation failed: {e}")
        return state

    async def _process_validation(self, state: WorkflowState) -> WorkflowState:
        """Process clinical validation"""
        if self.agents['validator'] and state.hypotheses:
            try:
                results = self.agents['validator'].validate_hypotheses(state.hypotheses)
                state.validations = results.get('validations', [])
            except Exception as e:
                state.errors.append(f"Clinical validation failed: {e}")
        return state

    async def _process_explanations(self, state: WorkflowState) -> WorkflowState:
        """Process XAI explanations"""
        if self.agents['xai'] and state.hypotheses:
            try:
                results = self.agents['xai'].generate_explanations(state.hypotheses)
                state.explanations = results.get('explanations', {})
            except Exception as e:
                state.errors.append(f"XAI explanations failed: {e}")
        return state

    async def _process_confidence(self, state: WorkflowState) -> WorkflowState:
        """Process confidence scoring"""
        if self.agents['confidence'] and state.hypotheses:
            try:
                results = self.agents['confidence'].compute_confidence(state.hypotheses)
                state.confidence_scores = results.get('confidence_scores', {})
                state.confidence_level = results.get('overall_confidence', 0.0)
            except Exception as e:
                state.errors.append(f"Confidence scoring failed: {e}")
        return state

    def _format_results(self, state: WorkflowState) -> Dict:
        """Format final results"""
        return {
            'patient_input': state.patient_input,
            'symptoms': state.symptoms,
            'hypotheses': state.hypotheses,
            'validations': state.validations,
            'explanations': state.explanations,
            'confidence_scores': state.confidence_scores,
            'final_diagnosis': state.final_diagnosis,
            'confidence_level': state.confidence_level,
            'recommendations': state.recommendations,
            'metadata': {
                'total_symptoms': len(state.symptoms),
                'total_hypotheses': len(state.hypotheses),
                'processing_time': state.total_duration,
                'errors': state.errors
            }
        }
