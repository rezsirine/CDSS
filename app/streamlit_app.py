import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import sys
from pathlib import Path
from orchestrator.workflow import ClinicalWorkflow

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))

def initialize_workflow():
    """Initialiser le workflow multi-agents"""
    try:
        # Utiliser un import direct pour éviter les problèmes circulaires
        from orchestrator.workflow import ClinicalWorkflow
        return ClinicalWorkflow(verbose=False, use_gpu=False)
    except Exception as e:
        st.error(f"Failed to load Clinical Workflow: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Clinical Decision Support System",
        page_icon="🩺",
        layout="wide"
    )
    
    # Style CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .symptom-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.markdown('<h1 class="main-header">🩺 Clinical Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown("### Multi-Agent System for Symptom Analysis")
    
    # Barre latérale
    with st.sidebar:
        st.header("Configuration")
        
        # Seuil de confiance
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.divider()
        
        # Exemples rapides
        st.header("Quick Examples")
        examples = {
            "Common Cold": "Patient presents with runny nose, sneezing, sore throat, and mild fever for 2 days.",
            "Migraine": "Severe headache with nausea, sensitivity to light and sound.",
            "Pneumonia": "High fever, productive cough with yellow sputum, chest pain."
        }
        
        selected_example = st.selectbox("Choose example:", list(examples.keys()))
        if selected_example:
            if st.button(f"Load: {selected_example}", use_container_width=True):
                st.session_state.example_text = examples[selected_example]
                st.rerun()
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Patient Symptoms")
        
        # Zone de texte
        text_input = st.text_area(
            "Describe symptoms in detail:",
            height=200,
            placeholder="Example: The 45-year-old male presents with fever (38.5°C), headache, and fatigue for 3 days.",
            key="symptom_input",
            value=st.session_state.get('example_text', '')
        )
        
        # Boutons d'action
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                for key in ['results', 'example_text', 'last_text']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Analyser les symptômes
        if analyze_btn and text_input.strip():
            with st.spinner("Running complete clinical analysis..."):
                try:
                    # Initialiser le workflow
                    if 'workflow' not in st.session_state:
                        st.session_state.workflow = initialize_workflow()
                    
                    if st.session_state.workflow:
                        # Exécuter le workflow
                        workflow_results = st.session_state.workflow.run_sync(text_input)
                        
                        # Sauvegarder les résultats
                        st.session_state.results = workflow_results
                        st.session_state.last_text = text_input
                        
                        st.success("Complete clinical analysis finished!")
                    else:
                        st.error("Workflow not available")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
    
    with col2:
        st.subheader("Quick Stats")
        
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            
            # Métriques
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Symptoms", results['metadata']['total_symptoms'])
            with metrics_col2:
                if 'confidence_level' in results:
                    st.metric("Confidence", f"{results['confidence_level']:.2f}")
            
            # Graphique des symptômes
            if results['symptoms']:
                df = pd.DataFrame(results['symptoms'])
                fig = px.bar(
                    df, 
                    x='symptom', 
                    y='confidence',
                    title='Symptom Confidence',
                    color='confidence',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter symptoms and click 'Analyze' to see results here.")
    
    # Afficher les résultats détaillés
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # Tab pour organiser l'affichage
        tab1, tab2, tab3 = st.tabs(["Symptoms", "Hypotheses", "Confidence"])
        
        with tab1:
            if results['symptoms']:
                st.subheader("Detected Symptoms")
                for i, symptom in enumerate(results['symptoms'], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {symptom['symptom'].upper()}**")
                            st.caption(f"Type: {symptom.get('type', 'N/A')} | Source: {symptom.get('source', 'N/A')}")
                        with col2:
                            st.metric("Confidence", f"{symptom.get('confidence', 0):.2f}")
                        with col3:
                            st.progress(symptom.get('confidence', 0))
                        st.divider()
        
        with tab2:
            if results.get('hypotheses'):
                st.subheader("Diagnostic Hypotheses")
                for i, hyp in enumerate(results['hypotheses'], 1):
                    if isinstance(hyp, dict):
                        with st.container():
                            st.markdown(f"**{i}. {hyp.get('diagnosis', 'Unknown')}**")
                            st.caption(f"Confidence: {hyp.get('confidence', 0):.2f}")
                            if 'explanation' in hyp:
                                st.info(hyp['explanation'])
                    else:
                        st.write(f"{i}. {hyp}")
        
        with tab3:
            if results.get('confidence_scores'):
                st.subheader("Confidence Analysis")
                conf_scores = results['confidence_scores']
                
                # Créer un DataFrame pour le graphique
                scores_df = pd.DataFrame({
                    'Metric': list(conf_scores.keys()),
                    'Value': list(conf_scores.values())
                })
                
                fig = px.bar(scores_df, x='Metric', y='Value', 
                           title='Confidence Score Components',
                           color='Value', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher les valeurs
                for key, value in conf_scores.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.3f}")

if __name__ == "__main__":
    # Initialiser l'état de session
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ''
    
    main()