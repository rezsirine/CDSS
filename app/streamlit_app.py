"""
Application Streamlit - Version simplifiée
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import sys
from pathlib import Path

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

# Configuration de la page
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .symptom-item {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_workflow():
    """Initialiser le workflow multi-agents complet"""
    try:
        from orchestrator.workflow import ClinicalWorkflow
        return ClinicalWorkflow(verbose=False)
    except Exception as e:
        st.error(f" Failed to load Clinical Workflow: {e}")
        return None

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🩺 Clinical Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown("### Multi-Agent System for Symptom Analysis")
    
    # Barre latérale
    with st.sidebar:
        st.header(" Configuration")
        
        # Seuil de confiance
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for symptoms"
        )
        
        st.divider()
        
        st.header(" System Info")
        st.metric("Workflow Status", "Ready" if 'workflow' in st.session_state else "Not Loaded")
        st.metric("Session", datetime.now().strftime("%H:%M:%S"))
        
        st.divider()
        
        # Exemples rapides
        st.header(" Quick Examples")
        examples = {
            "Common Cold": "Patient presents with runny nose, sneezing, sore throat, and mild fever for 2 days.",
            "Migraine": "Severe headache with nausea, sensitivity to light and sound.",
            "Pneumonia": "High fever, productive cough with yellow sputum, chest pain.",
            "Gastroenteritis": "Abdominal pain, diarrhea, vomiting for 24 hours."
        }
        
        selected_example = st.selectbox("Choose example:", list(examples.keys()))
        if selected_example:
            if st.button(f"Load: {selected_example}", use_container_width=True):
                st.session_state.example_text = examples[selected_example]
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Enter Patient Symptoms")
        
        # Zone de texte
        text_input = st.text_area(
            "Describe symptoms in detail:",
            height=200,
            placeholder="Example: The 45-year-old male presents with fever (38.5°C), headache, and fatigue for 3 days. No cough or shortness of breath.",
            key="symptom_input",
            value=st.session_state.get('example_text', '')
        )
        
        # Boutons d'action
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.pop('results', None)
            st.session_state.pop('example_text', None)
            st.rerun()
        
        # Analyser les symptômes avec le workflow complet
        if analyze_btn and text_input.strip():
            with st.spinner(" Running complete clinical analysis..."):
                try:
                    # Initialiser le workflow
                    if 'workflow' not in st.session_state:
                        st.session_state.workflow = initialize_workflow()

                    if st.session_state.workflow:
                        # Exécuter le workflow complet
                        workflow_results = st.session_state.workflow.run_sync(text_input)

                        # Sauvegarder dans la session
                        st.session_state.results = workflow_results
                        st.session_state.last_text = text_input

                        st.success(" Complete clinical analysis finished!")
                    else:
                        st.error(" Workflow not available")

                except Exception as e:
                    st.error(f" Error during analysis: {e}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
    
    with col2:
        st.subheader(" Quick Stats")
        
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            
            # Métriques
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Symptoms", results['metadata']['total_symptoms'])
            with metrics_col2:
                if results['symptoms']:
                    avg_conf = sum(s['confidence'] for s in results['symptoms']) / len(results['symptoms'])
                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
            
            # Graphique
            if results['symptoms']:
                df = pd.DataFrame(results['symptoms'])
                fig = px.bar(
                    df, 
                    x='symptom', 
                    y='confidence',
                    title='Symptom Confidence',
                    labels={'symptom': 'Symptom', 'confidence': 'Confidence'},
                    color='confidence',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(" Enter symptoms and click 'Analyze' to see results here.")
    
    # Afficher les résultats détaillés
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        if results['symptoms']:
            st.divider()
            st.subheader(" Detected Symptoms")
            
            for i, symptom in enumerate(results['symptoms'], 1):
                with st.container():
                    # Card pour chaque symptôme
                    confidence = symptom['confidence']
                    
                    # Couleur basée sur la confiance
                    if confidence >= 0.8:
                        color = "#10B981"  # Vert
                    elif confidence >= 0.5:
                        color = "#F59E0B"  # Orange
                    else:
                        color = "#EF4444"  # Rouge
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {symptom['symptom'].upper()}")
                        st.markdown(f"**Type:** {symptom['type']} | **Source:** {symptom['source']}")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}")
                    
                    with col3:
                        st.progress(confidence)
                    
                    # Contexte
                    if symptom.get('context'):
                        with st.expander("View context"):
                            st.info(symptom['context'])
                    
                    st.divider()
            
            # Options d'export
            with st.expander(" Export Results"):
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    # JSON
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label=" Download JSON",
                        data=json_str,
                        file_name=f"symptoms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col_exp2:
                    # CSV
                    df = pd.DataFrame(results['symptoms'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=" Download CSV",
                        data=csv,
                        file_name=f"symptoms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Texte original
        with st.expander(" View Original Text"):
            st.text(st.session_state.get('last_text', ''))

if __name__ == "__main__":
    # Initialiser l'état de session
    if 'results' not in st.session_state:
        st.session_state.results = None

    main()
