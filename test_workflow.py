#!/usr/bin/env python3
"""
Test du système amélioré
"""

import sys
from pathlib import Path

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent))

def test_enhanced_system():
    """Test du système avec les améliorations"""
    print("Testing Enhanced Clinical System")
    print("=" * 60)
    
    # Installer sacremoses si nécessaire
    try:
        import sacremoses
        print("✅ sacremoses installed")
    except ImportError:
        print("⚠️  Installing sacremoses...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sacremoses"])
        print("✅ sacremoses installed")
    
    try:
        from orchestrator.workflow import ClinicalWorkflow
        
        # Initialiser avec mode amélioré
        print("\nInitializing enhanced workflow...")
        workflow = ClinicalWorkflow(verbose=True, use_gpu=False)
        
        # Cas de test avec diagnostics attendus
        test_cases = [
            {
                "name": "Influenza-like illness",
                "input": "Patient has fever (38.5°C), dry cough, headache, muscle pain, and fatigue for 2 days.",
                "expected_diagnosis": "Influenza"
            },
            {
                "name": "Migraine case", 
                "input": "35-year-old female with severe throbbing headache on one side, nausea, vomiting, sensitivity to light and sound for 24 hours.",
                "expected_diagnosis": "Migraine"
            },
            {
                "name": "Pneumonia suspicion",
                "input": "68-year-old male with high fever (39°C), productive cough with green sputum, chest pain when breathing, shortness of breath, and fatigue for 5 days.",
                "expected_diagnosis": "Pneumonia"
            },
            {
                "name": "Gastroenteritis",
                "input": "Acute onset of nausea, vomiting, watery diarrhea, abdominal cramps, and low-grade fever for 12 hours.",
                "expected_diagnosis": "Gastroenteritis"
            }
        ]
        
        results_summary = []
        
        for case in test_cases:
            print(f"\n{'='*40}")
            print(f"Case: {case['name']}")
            print(f"Input: {case['input'][:80]}...")
            
            results = workflow.run_sync(case['input'])
            
            # Collecter les résultats
            case_results = {
                "case": case['name'],
                "symptoms": len(results['symptoms']),
                "hypotheses": len(results['hypotheses']),
                "confidence": results.get('confidence_level', 0),
                "top_diagnosis": results.get('final_diagnosis', 'Unknown'),
                "expected": case['expected_diagnosis']
            }
            
            print(f"  Symptoms detected: {case_results['symptoms']}")
            print(f"  Hypotheses generated: {case_results['hypotheses']}")
            print(f"  Confidence score: {case_results['confidence']:.3f}")
            print(f"  Top diagnosis: {case_results['top_diagnosis']}")
            print(f"  Expected: {case_results['expected']}")
            
            # Vérifier la correspondance
            if case['expected_diagnosis'].lower() in case_results['top_diagnosis'].lower():
                print(f"  ✅ Match with expected diagnosis!")
                case_results["match"] = True
            else:
                print(f"  ⚠️  No match with expected diagnosis")
                case_results["match"] = False
            
            results_summary.append(case_results)
            
            # Afficher les symptômes détectés
            if results['symptoms']:
                symptoms_list = [s['symptom'] for s in results['symptoms'][:5]]
                print(f"  Key symptoms: {', '.join(symptoms_list)}")
        
        # Résumé final
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        total_cases = len(results_summary)
        matches = sum(1 for r in results_summary if r.get('match', False))
        avg_confidence = sum(r['confidence'] for r in results_summary) / total_cases
        
        print(f"Total cases: {total_cases}")
        print(f"Correct matches: {matches}/{total_cases} ({matches/total_cases*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Détails par cas
        print(f"\nDetailed results:")
        for result in results_summary:
            match_icon = "✅" if result['match'] else "❌"
            print(f"  {match_icon} {result['case']}: {result['top_diagnosis']} "
                  f"(conf: {result['confidence']:.3f}, exp: {result['expected']})")
        
        if avg_confidence > 0.5 and matches > total_cases / 2:
            print(f"\n🎉 SYSTEM PERFORMANCE: GOOD")
            print("The system shows improved confidence and diagnostic accuracy.")
        else:
            print(f"\n⚠️  SYSTEM PERFORMANCE: NEEDS IMPROVEMENT")
            print("Consider further tuning of confidence scoring and validation.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("CLINICAL MULTI-AGENT SYSTEM - ENHANCED TEST")
    print("=" * 60)
    
    success = test_enhanced_system()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Enhanced system testing completed successfully!")
        print("\nNext steps:")
        print("1. Run Streamlit app: streamlit run app/streamlit_app.py")
        print("2. Test with real clinical cases")
        print("3. Consider adding more medical knowledge")
    else:
        print("\n❌ Testing failed. Please check the errors above.")

if __name__ == "__main__":
    main()