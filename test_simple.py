#!/usr/bin/env python3
"""
Test simple pour vérifier que le système fonctionne
"""

import sys
from pathlib import Path

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test des imports de base"""
    print("Testing basic imports...")
    
    tests = [
        ("data.ontologies", "MedicalOntology"),
        ("agents.symptom_agent", "SymptomAgent"),
        ("agents.hypothesis_agent", "HypothesisAgent"),
        ("agents.validator_agent", "ValidatorAgent"),
        ("agents.confidence_agent", "ConfidenceAgent"),
    ]
    
    all_passed = True
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"❌ {module_name}.{class_name} - not found")
                all_passed = False
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} - {e}")
            all_passed = False
    
    return all_passed

def test_ontology():
    """Test de l'ontologie médicale"""
    print("\nTesting Medical Ontology...")
    
    try:
        from data.ontologies import MedicalOntology
        
        ontology = MedicalOntology()
        
        # Test de cohérence
        result = ontology.check_consistency("influenza", ["fever", "cough", "fatigue"])
        print(f"Ontology check result: {result}")
        
        # Test de maladies liées
        related = ontology.get_related_diseases(["fever", "headache"])
        print(f"Related diseases: {len(related)} found")
        
        return True
    except Exception as e:
        print(f"ERROR in ontology test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validator():
    """Test du validateur"""
    print("\nTesting Validator Agent...")
    
    try:
        from agents.validator_agent import ValidatorAgent
        
        validator = ValidatorAgent(verbose=False)
        
        # Créer une hypothèse de test
        test_hypothesis = {
            "diagnosis": "influenza",
            "confidence": 0.8,
            "explanation": "Patient has fever and cough"
        }
        
        # Valider
        result = validator.validate_hypothesis(
            symptoms=["fever", "cough", "headache"],
            hypothesis=test_hypothesis
        )
        
        print(f"Validation result: Score={result.get('score', 0):.2f}, Valid={result.get('valid', False)}")
        
        return True
    except Exception as e:
        print(f"ERROR in validator test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 50)
    print("CLINICAL MULTI-AGENT SYSTEM - VALIDATION TEST")
    print("=" * 50)
    
    # Test 1: Imports
    imports_ok = test_basic_imports()
    
    # Test 2: Ontologie
    ontology_ok = test_ontology() if imports_ok else False
    
    # Test 3: Validateur
    validator_ok = test_validator() if imports_ok else False
    
    # Résumé
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Ontology: {'✅ PASS' if ontology_ok else '❌ FAIL'}")
    print(f"Validator: {'✅ PASS' if validator_ok else '❌ FAIL'}")
    
    if all([imports_ok, ontology_ok, validator_ok]):
        print("\n🎉 ALL TESTS PASSED! The system is ready.")
        print("\nNext steps:")
        print("1. python -m orchestrator.workflow")  # pour tester le workflow complet
        print("2. streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()