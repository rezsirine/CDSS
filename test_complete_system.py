"""
Test complet de tous les composants
"""

def test_all_components():
    """Test de tous les composants"""
    
    print("="*70)
    print("TESTING COMPLETE SYSTEM")
    print("="*70)
    
    all_tests_passed = True
    
    # Test 1: RAG
    print("\n[1/5] Testing RAG with PubMed...")
    try:
        from agents.hypothesis_agent_rag import HypothesisAgentWithRAG
        agent = HypothesisAgentWithRAG(verbose=False)
        hypotheses = agent.generate_hypotheses(["fever", "cough"], top_k=3)
        assert len(hypotheses) > 0
        print("  ✅ RAG working")
    except Exception as e:
        print(f"  ❌ RAG failed: {e}")
        all_tests_passed = False
    
    # Test 2: Datasets
    print("\n[2/5] Testing dataset loading...")
    try:
        from data.dataset_loader import MedicalDatasetLoader
        loader = MedicalDatasetLoader()
        datasets = loader.get_all_datasets()
        assert len(datasets) >= 3
        print(f"  ✅ Loaded {len(datasets)} datasets")
    except Exception as e:
        print(f"  ❌ Datasets failed: {e}")
        all_tests_passed = False
    
    # Test 3: Workflow
    print("\n[3/5] Testing complete workflow...")
    try:
        from orchestrator.workflow import ClinicalWorkflow
        workflow = ClinicalWorkflow(verbose=False)
        results = workflow.run_sync("Patient has fever and cough")
        assert 'hypotheses' in results
        assert 'confidence_level' in results
        print("  ✅ Workflow complete")
    except Exception as e:
        print(f"  ❌ Workflow failed: {e}")
        all_tests_passed = False
    
    # Test 4: Evaluation
    print("\n[4/5] Testing evaluation system...")
    try:
        from evaluation.evaluator import ClinicalEvaluator
        evaluator = ClinicalEvaluator(verbose=False)
        
        # Test diagnostic
        y_true = ["Influenza", "Pneumonia"]
        y_pred = ["Influenza", "Pneumonia"]
        y_proba = [0.8, 0.75]
        
        metrics = evaluator.evaluate_diagnostic(y_true, y_pred, y_proba)
        assert 'accuracy' in metrics
        print("  ✅ Evaluation working")
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        all_tests_passed = False
    
    # Test 5: All agents
    print("\n[5/5] Testing all agents...")
    try:
        from agents.symptom_agent import SymptomAgent
        from agents.hypothesis_agent_rag import HypothesisAgentWithRAG
        from agents.validator_agent import ValidatorAgent
        from agents.xai_agent import XAIAgent
        from agents.confidence_agent import ConfidenceAgent
        
        symptom_agent = SymptomAgent(verbose=False)
        hypothesis_agent = HypothesisAgentWithRAG(verbose=False)
        validator_agent = ValidatorAgent(verbose=False)
        xai_agent = XAIAgent(verbose=False)
        confidence_agent = ConfidenceAgent(verbose=False)
        
        print("  ✅ All agents initialized")
    except Exception as e:
        print(f"  ❌ Agents failed: {e}")
        all_tests_passed = False
    
    # Résultat final
    print("\n" + "="*70)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - SYSTEM READY!")
        print("\nYou can now:")
        print("  1. Run evaluation: python run_evaluation.py")
        print("  2. Launch Streamlit: streamlit run app/cdss_app.py")
        print("  3. Use as library: from orchestrator.workflow import ClinicalWorkflow")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("="*70)
    
    return all_tests_passed

if __name__ == "__main__":
    test_all_components()