#!/usr/bin/env python3
"""
Ajoute définitivement la méthode run_sync
"""

import sys
from pathlib import Path

def add_run_sync_method():
    """Ajoute la méthode run_sync manquante"""
    workflow_path = Path("orchestrator/workflow.py")
    
    if not workflow_path.exists():
        print(f" {workflow_path} not found")
        return False
    
    content = workflow_path.read_text(encoding='utf-8')
    
    # Vérifier si run_sync existe déjà
    if 'def run_sync' in content:
        print(" run_sync method already exists")
        return True
    
    print("Adding run_sync method to ClinicalWorkflow class...")
    
    # Trouver la fin de la classe ClinicalWorkflow
    lines = content.split('\n')
    new_lines = []
    in_clinical_workflow = False
    class_indent = 0
    added = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Détecter le début de la classe ClinicalWorkflow
        if 'class ClinicalWorkflow:' in line:
            in_clinical_workflow = True
            class_indent = len(line) - len(line.lstrip())
        
        # Détecter la fin de la classe (ligne vide après la dernière méthode)
        elif in_clinical_workflow and not added:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else 0
            
            # Si on trouve une ligne avec moins d'indentation que la classe
            # (fin de la classe) ou si c'est la dernière ligne
            if (line.strip() and current_indent < class_indent) or i == len(lines) - 1:
                # Ajouter run_sync juste avant
                run_sync_code = f'''{" " * (class_indent + 4)}def run_sync(self, patient_input: str):
{" " * (class_indent + 8)}\"\"\"
{" " * (class_indent + 8)}Exécute le workflow de manière synchrone
{" " * (class_indent + 8)}
{" " * (class_indent + 8)}Args:
{" " * (class_indent + 8)}    patient_input: Description textuelle des symptômes
{" " * (class_indent + 8)}
{" " * (class_indent + 8)}Returns:
{" " * (class_indent + 8)}    Dict avec les résultats du workflow
{" " * (class_indent + 8)}\"\"\"
{" " * (class_indent + 8)}import asyncio
{" " * (class_indent + 8)}try:
{" " * (class_indent + 12)}return asyncio.run(self.run(patient_input))
{" " * (class_indent + 8)}except RuntimeError:
{" " * (class_indent + 12)}# Si déjà dans une boucle d'événements (Streamlit, Jupyter, etc.)
{" " * (class_indent + 12)}loop = asyncio.new_event_loop()
{" " * (class_indent + 12)}asyncio.set_event_loop(loop)
{" " * (class_indent + 12)}try:
{" " * (class_indent + 16)}result = loop.run_until_complete(self.run(patient_input))
{" " * (class_indent + 16)}return result
{" " * (class_indent + 12)}finally:
{" " * (class_indent + 16)}loop.close()'''
                
                # Insérer avant la fin de la classe
                new_lines.insert(-1, run_sync_code)
                added = True
    
    if added:
        new_content = '\n'.join(new_lines)
        workflow_path.write_text(new_content, encoding='utf-8')
        print(" run_sync method added successfully!")
        
        # Vérifier
        if 'def run_sync' in new_content:
            print(" Verification passed: run_sync is in the file")
            return True
        else:
            print(" Verification failed: run_sync not found after adding")
            return False
    else:
        print(" Could not find where to insert run_sync method")
        return False

def test_run_sync():
    """Teste si run_sync fonctionne"""
    print("\nTesting run_sync method...")
    
    try:
        # Recharger le module
        import importlib
        import orchestrator.workflow
        
        importlib.reload(orchestrator.workflow)
        from orchestrator.workflow import ClinicalWorkflow
        
        workflow = ClinicalWorkflow(verbose=False)
        
        # Test simple
        results = workflow.run_sync("Patient has fever and cough")
        
        print(f" SUCCESS! run_sync works.")
        print(f"  Symptoms: {len(results['symptoms'])}")
        print(f"  Confidence: {results.get('confidence_level', 0):.2f}")
        
        return True
        
    except AttributeError as e:
        print(f" AttributeError: {e}")
        return False
    except Exception as e:
        print(f" Other error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("FIXING 'run_sync' METHOD ERROR")
    print("=" * 60)
    
    # 1. Ajouter la méthode
    success = add_run_sync_method()
    
    if success:
        # 2. Tester
        test_success = test_run_sync()
        
        if test_success:
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! System is now fully operational!")
            print("\nYou can now:")
            print("1. Run: python test_workflow.py")
            print("2. Launch: streamlit run app/streamlit_app.py")
            print("3. BioGPT-Large is ready (6.29GB downloaded)")
        else:
            print("\n Method added but test failed")
    else:
        print("\n Failed to add run_sync method")

if __name__ == "__main__":
    main()