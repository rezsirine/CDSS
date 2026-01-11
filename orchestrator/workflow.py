"""
orchestrator/workflow.py
Orchestration du workflow multi-agents clinique
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import traceback

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
        
        Args:
            verbose: Mode verbeux
            use_gpu: Utiliser GPU si disponible
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        
        if verbose:
            print("Initializing Clinical Workflow...")
            print("=" * 50)
        
        # Initialiser les agents
        self.agents = {}
        self._initialize_agents()
        
        if verbose:
            print("=" * 50)
            print("Clinical Workflow initialized")
            print(f"Agents loaded: {len(self.agents)}")
            for agent_name, agent in self.agents.items():
                status = "✅" if agent is not None else "❌"
                print(f"  {status} {agent_name}")
    
    def _initialize_agents(self):
        """Initialise tous les agents"""
        try:
            # Symptom Agent
            if self.verbose:
                print("Loading SymptomAgent...")
            try:
                from agents.symptom_agent import SymptomAgent
                self.agents['symptom'] = SymptomAgent(verbose=self.verbose)
            except ImportError as e:
                if self.verbose:
                    print(f"  ⚠️ SymptomAgent import failed: {e}")
                self.agents['symptom'] = self._create_dummy_agent("SymptomAgent")
            
            # Hypothesis Agent
            if self.verbose:
                print("Loading HypothesisAgent...")
            try:
                from agents.hypothesis_agent import HypothesisAgent
                self.agents['hypothesis'] = HypothesisAgent(
                    use_gpu=self.use_gpu, 
                    verbose=self.verbose
                )
            except ImportError as e:
                if self.verbose:
                    print(f"  ⚠️ HypothesisAgent import failed: {e}")
                self.agents['hypothesis'] = self._create_dummy_agent("HypothesisAgent")
            
            # Validator Agent
            if self.verbose:
                print("Loading ValidatorAgent...")
            try:
                from agents.validator_agent import ValidatorAgent
                self.agents['validator'] = ValidatorAgent(verbose=self.verbose)
            except ImportError as e:
                if self.verbose:
                    print(f"  ⚠️ ValidatorAgent import failed: {e}")
                self.agents['validator'] = self._create_dummy_agent("ValidatorAgent")
            
            # XAI Agent
            if self.verbose:
                print("Loading XAIAgent...")
            try:
                from agents.xai_agent import XAIAgent
                self.agents['xai'] = XAIAgent(verbose=self.verbose)
            except ImportError as e:
                if self.verbose:
                    print(f"  ⚠️ XAIAgent import failed: {e}")
                self.agents['xai'] = self._create_dummy_agent("XAIAgent")
            
            # Confidence Agent
            if self.verbose:
                print("Loading ConfidenceAgent...")
            try:
                from agents.confidence_agent import ConfidenceAgent
                self.agents['confidence'] = ConfidenceAgent(verbose=self.verbose)
            except ImportError as e:
                if self.verbose:
                    print(f"  ⚠️ ConfidenceAgent import failed: {e}")
                self.agents['confidence'] = self._create_dummy_agent("ConfidenceAgent")
                
        except Exception as e:
            if self.verbose:
                print(f"Fatal error during agent initialization: {e}")
            raise
    
    def _create_dummy_agent(self, name: str):
        """Crée un agent factice pour le développement"""
        class DummyAgent:
            def __init__(self, agent_name):
                self.name = agent_name
                self.dummy_data = {
                    "status": "dummy",
                    "message": f"{agent_name} is in dummy mode"
                }
            
            def extract_from_text(self, text):
                return {"symptoms": [
                    {"symptom": "fever", "confidence": 0.8, "type": "SYMPTOM", "source": "dummy"},
                    {"symptom": "headache", "confidence": 0.7, "type": "SYMPTOM", "source": "dummy"}
                ]}
            
            def generate_hypotheses(self, symptoms, top_k=5):
                return [
                    {"diagnosis": "Influenza", "confidence": 0.75, "explanation": "Common viral infection"},
                    {"diagnosis": "Migraine", "confidence": 0.65, "explanation": "Neurological headache"}
                ]
            
            def validate_hypotheses(self, hypotheses):
                return [{"validation": {"score": 0.7, "valid": True}} for _ in hypotheses]
            
            def explain_hypothesis(self, symptoms, hypothesis):
                return {
                    "hypothesis": hypothesis.get('diagnosis', 'Unknown'),
                    "explanations": {"method": "dummy", "key_points": ["Explanation not available"]}
                }
            
            def compute_confidence_score(self, **kwargs):
                return {
                    "composite_confidence": 0.7,
                    "epistemic_uncertainty": 0.3,
                    "clinical_consistency": 0.8,
                    "explainability_score": 0.6
                }
        
        if self.verbose:
            print(f"  ⚠️ Created dummy agent for {name}")
        return DummyAgent(name)
    
    async def run(self, patient_input: str) -> Dict:
        """
        Exécute le workflow complet de manière asynchrone
        
        Args:
            patient_input: Description textuelle des symptômes
            
        Returns:
            Dict avec les résultats du workflow
        """
        # Initialiser l'état
        state = WorkflowState(patient_input=patient_input)
        state.start_time = datetime.now()
        
        if self.verbose:
            print(f"\nStarting workflow for input: {patient_input[:100]}...")
        
        try:
            # Étape 1: Extraction des symptômes
            state = await self._process_symptoms(state)
            
            # Étape 2: Génération d'hypothèses
            state = await self._process_hypotheses(state)
            
            # Étape 3: Validation clinique
            state = await self._process_validation(state)
            
            # Étape 4: Explications XAI
            state = await self._process_explanations(state)
            
            # Étape 5: Calcul de confiance
            state = await self._process_confidence(state)
            
            # Étape 6: Synthèse finale
            state = await self._process_finalization(state)
            
        except Exception as e:
            error_msg = f"Workflow execution error: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
                traceback.print_exc()
            state.errors.append(error_msg)
        
        # Finalisation
        state.end_time = datetime.now()
        if state.start_time:
            state.total_duration = (state.end_time - state.start_time).total_seconds()
        
        if self.verbose:
            status = "✅" if not state.errors else "⚠️"
            print(f"\n{status} Workflow completed in {state.total_duration:.2f} seconds")
            if state.errors:
                print(f"Errors: {len(state.errors)}")
        
        return self._format_results(state)
    
    async def _process_symptoms(self, state: WorkflowState) -> WorkflowState:
        """Traite l'extraction des symptômes"""
        if self.verbose:
            print("\n[Step 1] Extracting symptoms...")
        
        if self.agents['symptom']:
            try:
                # Exécuter de manière asynchrone
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self.agents['symptom'].extract_from_text,
                    state.patient_input
                )
                
                state.symptoms = results.get('symptoms', [])
                
                if self.verbose:
                    print(f"  ✅ Extracted {len(state.symptoms)} symptoms")
                    for symptom in state.symptoms[:5]:  # Afficher les 5 premiers
                        print(f"    - {symptom['symptom']} ({symptom.get('confidence', 0):.2f})")
                    if len(state.symptoms) > 5:
                        print(f"    ... and {len(state.symptoms) - 5} more")
                        
            except Exception as e:
                error_msg = f"Symptom extraction failed: {str(e)}"
                state.errors.append(error_msg)
                if self.verbose:
                    print(f"  ❌ {error_msg}")
        else:
            error_msg = "Symptom agent not available"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
        
        return state
    
    async def _process_hypotheses(self, state: WorkflowState) -> WorkflowState:
        """Traite la génération d'hypothèses"""
        if not state.symptoms:
            if self.verbose:
                print("\n[Step 2] Skipping hypothesis generation (no symptoms)")
            return state
        
        if self.verbose:
            print("\n[Step 2] Generating diagnostic hypotheses...")
        
        if self.agents['hypothesis']:
            try:
                # Extraire les textes de symptômes
                symptom_texts = [s['symptom'] for s in state.symptoms]
                
                # Générer les hypothèses
                results = self.agents['hypothesis'].generate_hypotheses(
                    symptom_texts, 
                    top_k=5
                )
                
                # Formater les résultats
                if isinstance(results, list):
                    state.hypotheses = results
                elif isinstance(results, dict) and 'hypotheses' in results:
                    state.hypotheses = results['hypotheses']
                else:
                    state.hypotheses = results
                
                if self.verbose:
                    print(f"  ✅ Generated {len(state.hypotheses)} hypotheses")
                    for i, hyp in enumerate(state.hypotheses[:3], 1):
                        if isinstance(hyp, dict):
                            diag = hyp.get('diagnosis', 'Unknown')
                            conf = hyp.get('confidence', 0)
                            print(f"    {i}. {diag} ({conf:.2f})")
                        else:
                            print(f"    {i}. {str(hyp)[:50]}...")
                            
            except Exception as e:
                error_msg = f"Hypothesis generation failed: {str(e)}"
                state.errors.append(error_msg)
                if self.verbose:
                    print(f"  ❌ {error_msg}")
        else:
            error_msg = "Hypothesis agent not available"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
        
        return state
    
    async def _process_validation(self, state: WorkflowState) -> WorkflowState:
        """Traite la validation clinique"""
        if not state.hypotheses:
            if self.verbose:
                print("\n[Step 3] Skipping validation (no hypotheses)")
            return state
        
        if self.verbose:
            print("\n[Step 3] Validating hypotheses clinically...")
        
        if self.agents['validator']:
            try:
                # Extraire les symptômes pour la validation
                symptom_texts = [s['symptom'] for s in state.symptoms]
                
                # Valider chaque hypothèse
                validated_hypotheses = []
                for hypothesis in state.hypotheses:
                    if isinstance(hypothesis, dict):
                        validation_result = self.agents['validator'].validate_hypothesis(
                            symptom_texts,
                            hypothesis
                        )
                        
                        # Ajouter la validation à l'hypothèse
                        validated_hyp = {
                            **hypothesis,
                            "validation": validation_result
                        }
                        validated_hypotheses.append(validated_hyp)
                
                # Mettre à jour les hypothèses avec validation
                if validated_hypotheses:
                    state.hypotheses = validated_hypotheses
                    state.validations = [h.get('validation', {}) for h in validated_hypotheses]
                
                if self.verbose:
                    valid_count = sum(1 for v in state.validations if v.get('valid', False))
                    print(f"  ✅ Validated {len(state.validations)} hypotheses ({valid_count} valid)")
                    
            except Exception as e:
                error_msg = f"Clinical validation failed: {str(e)}"
                state.errors.append(error_msg)
                if self.verbose:
                    print(f"  ❌ {error_msg}")
        else:
            error_msg = "Validator agent not available"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
        
        return state
    
    async def _process_explanations(self, state: WorkflowState) -> WorkflowState:
        """Traite la génération d'explications XAI"""
        if not state.hypotheses:
            if self.verbose:
                print("\n[Step 4] Skipping explanations (no hypotheses)")
            return state
        
        if self.verbose:
            print("\n[Step 4] Generating XAI explanations...")
        
        if self.agents['xai']:
            try:
                symptom_texts = [s['symptom'] for s in state.symptoms]
                explanations = {}
                
                # Générer des explications pour chaque hypothèse
                for i, hypothesis in enumerate(state.hypotheses[:3]):  # Limiter aux 3 premières
                    if isinstance(hypothesis, dict):
                        explanation = self.agents['xai'].explain_hypothesis(
                            symptom_texts,
                            hypothesis
                        )
                        explanations[f"hypothesis_{i}"] = explanation
                
                state.explanations = explanations
                
                if self.verbose:
                    print(f"  ✅ Generated explanations for {len(explanations)} hypotheses")
                    
            except Exception as e:
                error_msg = f"XAI explanation failed: {str(e)}"
                state.errors.append(error_msg)
                if self.verbose:
                    print(f"  ❌ {error_msg}")
        else:
            error_msg = "XAI agent not available"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
        
        return state
    
async def _process_confidence(self, state: WorkflowState) -> WorkflowState:
    """Traite le calcul de confiance amélioré"""
    if not state.hypotheses:
        if self.verbose:
            print("\n[Step 5] Skipping confidence calculation (no hypotheses)")
        return state
    
    if self.verbose:
        print("\n[Step 5] Computing confidence scores...")
    
    if self.agents['confidence']:
        try:
            # Préparer les données pour le calcul de confiance
            hypothesis_probs = {}
            validation_scores = {}
            xai_scores = {}
            
            # Extraire les scores des hypothèses
            for i, hypothesis in enumerate(state.hypotheses):
                key = f"hypothesis_{i}"
                
                # Probabilités des hypothèses
                hypothesis_probs[key] = hypothesis.get('confidence', 0.6)  # Augmenté de 0.5
                
                # Scores de validation
                if 'validation' in hypothesis:
                    val_score = hypothesis['validation'].get('score', 0.5)
                    # Ajuster si validation est "valid"
                    if hypothesis['validation'].get('valid', False):
                        val_score = max(val_score, 0.7)
                    validation_scores[key] = val_score
                else:
                    validation_scores[key] = 0.5
                
                # Scores XAI
                xai_key = f"hypothesis_{i}"
                if xai_key in state.explanations:
                    xai_scores[key] = 0.7  # Augmenté
                else:
                    xai_scores[key] = 0.5
            
            # Calculer les scores de confiance avec nombre de symptômes
            confidence_results = self.agents['confidence'].compute_confidence_score(
                hypothesis_probs=hypothesis_probs,
                validation_scores=validation_scores,
                xai_scores=xai_scores,
                symptoms_count=len(state.symptoms)
            )
            
            state.confidence_scores = confidence_results
            state.confidence_level = confidence_results.get('composite_confidence', 0.0)
            
            if self.verbose:
                conf_level = state.confidence_level
                if conf_level >= 0.7:
                    confidence_emoji = "🔵"
                elif conf_level >= 0.5:
                    confidence_emoji = "🟢"
                elif conf_level >= 0.3:
                    confidence_emoji = "🟡"
                else:
                    confidence_emoji = "🔴"
                
                print(f"  {confidence_emoji} Confidence level: {state.confidence_level:.3f}")
                print(f"    Top hypothesis: {confidence_results.get('top_hypothesis_prob', 0):.3f}")
                print(f"    Clinical: {confidence_results.get('clinical_consistency', 0):.3f}")
                print(f"    Symptoms: {len(state.symptoms)}")
                
        except Exception as e:
            error_msg = f"Confidence calculation failed: {str(e)}"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
    else:
        error_msg = "Confidence agent not available"
        state.errors.append(error_msg)
        if self.verbose:
            print(f"  ❌ {error_msg}")
    
    return state
    async def _process_finalization(self, state: WorkflowState) -> WorkflowState:
        """Traite la finalisation et les recommandations"""
        if self.verbose:
            print("\n[Step 6] Finalizing results...")
        
        try:
            # Déterminer le diagnostic final (hypothèse avec score de validation le plus élevé)
            if state.hypotheses and state.validations:
                best_hypothesis = max(
                    zip(state.hypotheses, state.validations),
                    key=lambda x: x[1].get('score', 0)
                )
                state.final_diagnosis = best_hypothesis[0].get('diagnosis', 'Unknown')
            
            # Générer des recommandations
            state.recommendations = self._generate_recommendations(state)
            
            if self.verbose:
                if state.final_diagnosis:
                    print(f"  ✅ Final diagnosis: {state.final_diagnosis}")
                print(f"  ✅ Generated {len(state.recommendations)} recommendations")
                
        except Exception as e:
            error_msg = f"Finalization failed: {str(e)}"
            state.errors.append(error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
        
        return state
    
    def _generate_recommendations(self, state: WorkflowState) -> List[str]:
        """Génère des recommandations cliniques basées sur les résultats"""
        recommendations = []
        
        # Recommandations basées sur la confiance
        if state.confidence_level < 0.5:
            recommendations.append("Consider additional tests due to low diagnostic confidence")
        
        # Recommandations basées sur les symptômes
        if state.symptoms:
            symptom_count = len(state.symptoms)
            if symptom_count < 3:
                recommendations.append("Gather more detailed symptom history")
        
        # Recommandations basées sur les drapeaux rouges
        for validation in state.validations:
            flags = validation.get('flags', [])
            for flag in flags:
                if 'emergency' in flag.lower() or 'urgent' in flag.lower():
                    recommendations.append("Consider urgent evaluation")
                    break
        
        # Recommandations génériques
        recommendations.extend([
            "Review patient medical history",
            "Consider differential diagnoses",
            "Monitor symptom progression"
        ])
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    def _format_results(self, state: WorkflowState) -> Dict:
        """Formate les résultats finaux"""
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
                'start_time': state.start_time.isoformat() if state.start_time else None,
                'end_time': state.end_time.isoformat() if state.end_time else None,
                'errors': state.errors,
                'agent_status': {
                    name: "available" if agent else "unavailable"
                    for name, agent in self.agents.items()
                }
            }
        }
    
    def run_sync(self, patient_input: str) -> Dict:
        """
        Exécute le workflow de manière synchrone
        
        Args:
            patient_input: Description textuelle des symptômes
            
        Returns:
            Dict avec les résultats du workflow
        """
        try:
            # Essayer d'exécuter dans la boucle d'événements courante
            return asyncio.run(self.run(patient_input))
        except RuntimeError:
            # Si déjà dans une boucle d'événements (Streamlit, Jupyter, etc.)
            # Créer une nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.run(patient_input))
                return result
            finally:
                loop.close()


# Fonction utilitaire pour tester directement le workflow
def test_workflow():
    """Teste le workflow directement"""
    import sys
    from pathlib import Path
    
    # Ajouter le chemin parent pour les imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    print("Testing Clinical Workflow...")
    print("=" * 60)
    
    # Cas de test
    test_cases = [
        "Patient has fever and cough for 3 days.",
        "45-year-old female with headache, nausea, and sensitivity to light.",
        "High fever, productive cough with yellow sputum, chest pain."
    ]
    
    workflow = ClinicalWorkflow(verbose=True)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case}")
        
        results = workflow.run_sync(test_case)
        
        print(f"  Symptoms: {len(results['symptoms'])}")
        print(f"  Hypotheses: {len(results['hypotheses'])}")
        print(f"  Confidence: {results.get('confidence_level', 0):.2f}")
        
        if results.get('final_diagnosis'):
            print(f"  Final diagnosis: {results['final_diagnosis']}")
        
        print("-" * 40)
    def run_sync(self, patient_input: str) -> Dict:
        """
        Exécute le workflow de manière synchrone
        
        Args:
            patient_input: Description textuelle des symptômes
            
        Returns:
            Dict avec les résultats du workflow
        """
        try:
            # Essayer d'exécuter dans la boucle d'événements courante
            return asyncio.run(self.run(patient_input))
        except RuntimeError:
            # Si déjà dans une boucle d'événements (Streamlit, Jupyter, etc.)
            # Créer une nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.run(patient_input))
                return result
            finally:
                loop.close()

if __name__ == "__main__":
    # Exécuter le test si le fichier est exécuté directement
    test_workflow()