"""
Test simple du système
"""

import sys
from pathlib import Path

def test_basic():
    """Test basique des imports"""
    print(" Testing Basic Imports")
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(" spaCy: OK")
    except Exception as e:
        print(f" spaCy: {e}")
    
    # Test transformers
    try:
        from transformers import pipeline
        print(" Transformers: OK")
    except Exception as e:
        print(f" Transformers: {e}")
    
    # Test SymptomAgent
    try:
        sys.path.append(str(Path(__file__).parent))
        from agents.symptom_agent import SymptomAgent
        print(" SymptomAgent: OK")
    except Exception as e:
        print(f" SymptomAgent: {e}")
    
    # Test Streamlit
    try:
        import streamlit as st
        print(" Streamlit: OK")
    except Exception as e:
        print(f" Streamlit: {e}")

def test_agent_functionality():
    """Tester la fonctionnalité de l'agent"""
    print("\n Testing Agent Functionality")
    
    try:
        from agents.symptom_agent import SymptomAgent
        
        agent = SymptomAgent(verbose=False)
        text = "The patient has fever and headache for 2 days."
        
        print(f"Input text: {text}")
        results = agent.extract_from_text(text)
        
        print(f"\nFound {results['metadata']['total_symptoms']} symptom(s):")
        for symptom in results['symptoms']:
            print(f"  - {symptom['symptom']} (confidence: {symptom['confidence']:.2f}, source: {symptom['source']})")
        
        return True
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(" CLINICAL MULTI-AGENT SYSTEM TEST")
    
    # Test des imports
    test_basic()
    
    # Test de l'agent
    success = test_agent_functionality()
    
    if success:
        print(" SUCCESS! System is working.")
        print("\n Next steps:")
        print("1. Run: streamlit run app/streamlit_app.py")
        print("2. Open: http://localhost:8501")
        print("3. Start testing with clinical cases!")
    else:
        print(" Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()