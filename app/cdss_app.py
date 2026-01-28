# app/cdss_app.py - Clinical Decision Support System
import sys
import os
from pathlib import Path
import streamlit as st
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION DES CHEMINS - CORRECTION CRITIQUE
# ============================================================================

# CHEMIN ABSOLU - C'est la correction principale
current_file = Path(__file__).resolve()  # Chemin ABSOLU du fichier
current_dir = current_file.parent        # app/ (absolu)
project_root = current_dir.parent        # CDSS/ (absolu)

# Afficher pour débogage (optionnel, peut être retiré après)
print(f" current_file: {current_file}")
print(f" current_dir: {current_dir}")
print(f" project_root: {project_root}")

# Ajouter les chemins ABSOLUS pour les imports
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestrator"))
sys.path.insert(0, str(project_root / "agents"))

# Désactiver les téléchargements automatiques
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ============================================================================
# FONCTIONS UTILITAIRES - IDENTIQUES À VOTRE CODE
# ============================================================================

def init_workflow():
    """Initialise le workflow clinique"""
    try:
        from orchestrator.workflow import ClinicalWorkflow
        workflow = ClinicalWorkflow(verbose=True, use_gpu=False)
        st.success(" Système clinique initialisé avec succès")
        return workflow
    except ImportError as e:
        st.error(f" Erreur d'import: {e}")
        st.info(f"Vérifiez que le fichier 'orchestrator/workflow.py' existe dans: {project_root}")
        return None
    except Exception as e:
        st.error(f" Erreur d'initialisation: {e}")
        return None

def analyze_symptoms(workflow, text):
    """Analyse les symptômes du patient"""
    if not text.strip():
        return {"error": "Veuillez entrer des symptômes"}
    
    try:
        with st.spinner(" Analyse en cours..."):
            results = workflow.run_sync(text)
            return results
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# FONCTION PRINCIPALE - IDENTIQUE À VOTRE CODE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Clinical Decision Support System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .symptom-card {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ef4444 0%, #f97316 50%, #22c55e 100%);
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Decision Support System</h1>
        <p>Multi-Agent AI System for Medical Symptom Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'état de session
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'symptom_text' not in st.session_state:
        st.session_state.symptom_text = ""
    
    # ============================================================================
    # BARRE LATÉRALE
    # ============================================================================
    with st.sidebar:
        st.header("⚙️ Configuration du système")
        
        # Bouton d'initialisation
        if st.button(" Initialiser le système", type="primary", use_container_width=True):
            with st.spinner("Chargement des modèles IA..."):
                st.session_state.workflow = init_workflow()
        
        st.divider()
        
        # Informations système
        st.header(" Informations")
        if st.session_state.workflow:
            st.success(" Système actif")
        else:
            st.warning(" Système non initialisé")
        
        st.divider()
        
        # Exemples prédéfinis
        st.header(" Exemples rapides")
        
        examples = {
            "Rhume commun": "Patient présente un nez qui coule, éternuements, mal de gorge et fièvre légère depuis 2 jours.",
            "Grippe": "Fièvre élevée (39°C), frissons, courbatures musculaires, fatigue intense, maux de tête.",
            "Migraine": "Mal de tête sévère unilatéral, nausées, vomissements, sensibilité à la lumière et au bruit.",
            "COVID-19": "Fièvre, toux sèche persistante, fatigue, perte du goût et de l'odorat, essoufflement.",
            "Pneumonie": "Fièvre élevée, toux productive avec expectorations jaunes/vertes, douleur thoracique, frissons.",
            "Gastroentérite": "Nausées, vomissements, diarrhée, crampes abdominales, fièvre modérée."
        }
        
        selected_example = st.selectbox("Choisir un exemple:", list(examples.keys()))
        
        if st.button(" Charger cet exemple", use_container_width=True):
            st.session_state.symptom_text = examples[selected_example]
            st.rerun()
        
        st.divider()
        
        # Aide
        with st.expander(" Conseils d'utilisation"):
            st.markdown("""
            **Pour de meilleurs résultats:**
            1. Soyez précis dans la description
            2. Mentionnez la durée des symptômes
            3. Incluez les signes vitaux (température, etc.)
            4. Notez les antécédents médicaux pertinents
            5. Mentionnez les médicaments actuels
            """)
    
    # ============================================================================
    # SECTION PRINCIPALE - ENTRÉE DES SYMPTÔMES
    # ============================================================================
    st.header(" Description des symptômes du patient")
    
    # Zone de texte pour l'entrée
    symptom_text = st.text_area(
        "Décrivez en détail les symptômes, signes cliniques et antécédents:",
        height=180,
        value=st.session_state.symptom_text,
        placeholder="Exemple: Patient de 45 ans, masculin, présente une fièvre à 38.5°C depuis 3 jours, accompagnée d'une toux sèche, de fatigue et de maux de tête. Pas d'antécédents médicaux significatifs. Aucun médicament actuellement.",
        key="main_input"
    )
    
    # Boutons d'action
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyze_btn = st.button(
            " Analyser les symptômes",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.workflow is None
        )
    
    with col2:
        if st.button(" Effacer", use_container_width=True):
            st.session_state.symptom_text = ""
            st.session_state.results = None
            st.rerun()
    
    with col3:
        if st.button(" Historique", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
            st.rerun()
    
    # Message si le système n'est pas initialisé
    if st.session_state.workflow is None:
        st.warning("""
         **Le système n'est pas initialisé**
        
        Pour commencer l'analyse:
        1. Cliquez sur **'Initialiser le système'** dans la barre latérale
        2. Attendez que tous les modèles soient chargés
        3. Entrez les symptômes ci-dessus
        """)
        
        # Informations de débogage
        with st.expander(" Informations techniques"):
            st.write(f"**Répertoire du projet:** {project_root}")
            st.write(f"**Répertoire actuel:** {current_dir}")
            st.write(f"**Chemin Python:** {sys.path[:3]}")
            
            # Vérifier l'existence des fichiers
            workflow_path = project_root / "orchestrator" / "workflow.py"
            st.write(f"**Fichier workflow:** {workflow_path}")
            st.write(f"**Existe:** {workflow_path.exists()}")
            
            # Vérifier aussi les __init__.py
            init_orchestrator = project_root / "orchestrator" / "__init__.py"
            st.write(f"**__init__.py orchestrator:** {init_orchestrator.exists()}")
            init_agents = project_root / "agents" / "__init__.py"
            st.write(f"**__init__.py agents:** {init_agents.exists()}")
        
        return
    
    # ============================================================================
    # ANALYSE DES SYMPTÔMES
    # ============================================================================
    if analyze_btn and symptom_text.strip():
        # Sauvegarder le texte
        st.session_state.symptom_text = symptom_text
        
        # Exécuter l'analyse
        results = analyze_symptoms(st.session_state.workflow, symptom_text)
        
        if "error" in results:
            st.error(f" Erreur d'analyse: {results['error']}")
        else:
            st.session_state.results = results
            st.session_state.last_analysis = datetime.now()
            
            # Afficher un message de succès
            processing_time = results.get('processing_time', 0)
            st.success(f" Analyse terminée en {processing_time:.2f} secondes")
    
    # ============================================================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================================================
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown("---")
        st.header(" Résultats de l'analyse clinique")
        
        # Créer des onglets pour organiser les résultats
        tab1, tab2, tab3, tab4 = st.tabs([
            " Symptômes détectés",
            "🤔 Hypothèses diagnostiques", 
            "📈 Analyse de confiance",
            "💾 Export des résultats"
        ])
        
        # ============================================
        # TAB 1: SYMPTÔMES DÉTECTÉS
        # ============================================
        with tab1:
            if results.get('symptoms'):
                st.subheader(f"Symptômes détectés ({len(results['symptoms'])})")
                
                for i, symptom in enumerate(results['symptoms'], 1):
                    with st.container():
                        col_a, col_b, col_c = st.columns([3, 1, 2])
                        
                        with col_a:
                            st.markdown(f"**{i}. {symptom.get('name', 'Symptôme').upper()}**")
                            if symptom.get('type'):
                                st.caption(f"Type: {symptom['type']}")
                        
                        with col_b:
                            confidence = symptom.get('confidence', 0)
                            st.metric("Confiance", f"{confidence:.2f}")
                        
                        with col_c:
                            st.progress(confidence)
                        
                        # Détails supplémentaires
                        if symptom.get('details'):
                            with st.expander(" Détails"):
                                st.write(symptom['details'])
                        
                        st.divider()
            else:
                st.info(" Aucun symptôme spécifique n'a été détecté dans la description.")
        
        # ============================================
        # TAB 2: HYPOTHÈSES DIAGNOSTIQUES
        # ============================================
        with tab2:
            if results.get('hypotheses'):
                st.subheader("Hypothèses diagnostiques classées par probabilité")
                
                for i, hypothesis in enumerate(results['hypotheses'], 1):
                    score = hypothesis.get('score', 0)
                    diagnosis = hypothesis.get('diagnosis', 'Diagnostic inconnu')
                    
                    # Créer une carte pour chaque hypothèse
                    with st.container():
                        # En-tête de l'hypothèse
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"### {i}. {diagnosis}")
                        
                        with col2:
                            # Barre de progression colorée
                            if score >= 0.7:
                                color = "🟢"
                            elif score >= 0.4:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.metric(f"Score {color}", f"{score:.3f}")
                        
                        # Explication
                        if hypothesis.get('explanation'):
                            st.info(f"**Explication:** {hypothesis['explanation']}")
                        
                        # Éléments de support
                        if hypothesis.get('supporting_evidence'):
                            with st.expander(" Éléments de support"):
                                for evidence in hypothesis['supporting_evidence']:
                                    st.write(f"• {evidence}")
                        
                        # Éléments contradictoires
                        if hypothesis.get('contradicting_evidence'):
                            with st.expander(" Éléments contradictoires"):
                                for evidence in hypothesis['contradicting_evidence']:
                                    st.write(f"• {evidence}")
                        
                        # Recommandations
                        if hypothesis.get('recommendations'):
                            with st.expander("💡 Recommandations"):
                                for rec in hypothesis['recommendations']:
                                    st.write(f"• {rec}")
                        
                        st.divider()
            else:
                st.info(" Aucune hypothèse diagnostique n'a pu être générée.")
        
        # ============================================
        # TAB 3: ANALYSE DE CONFIANCE
        # ============================================
        with tab3:
            st.subheader("Analyse de la fiabilité des résultats")
            
            # Score de confiance global
            if 'confidence_level' in results:
                confidence = results['confidence_level']
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.metric("Niveau de confiance global", f"{confidence:.3f}")
                
                with col_b:
                    if confidence >= 0.8:
                        st.success("🟢 Très fiable")
                    elif confidence >= 0.6:
                        st.warning("🟡 Modérément fiable")
                    else:
                        st.error("🔴 Fiabilité limitée")
                
                # Barre de progression
                st.progress(confidence)
            
            # Scores détaillés
            if results.get('confidence_scores'):
                st.subheader("Composantes de la confiance")
                
                scores = results['confidence_scores']
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        col1, col2, col3 = st.columns([2, 1, 3])
                        
                        with col1:
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                        
                        with col2:
                            st.write(f"{value:.3f}")
                        
                        with col3:
                            st.progress(value)
            
            # Avertissements
            if confidence < 0.5:
                st.warning("""
                 **Attention:** La confiance dans ces résultats est limitée.
                
                **Considérations:**
                - Les résultats doivent être interprétés avec prudence
                - Consulter un médecin pour confirmation
                - Fournir plus de détails cliniques si possible
                """)
        
        # ============================================
        # TAB 4: EXPORT DES RÉSULTATS
        # ============================================
        with tab4:
            st.subheader("Exporter les résultats de l'analyse")
            
            # Format JSON
            st.markdown("#### 📄 Format JSON (Complet)")
            json_data = json.dumps(results, indent=2, default=str, ensure_ascii=False)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(json_data[:1000] + ("..." if len(json_data) > 1000 else ""), language='json')
            
            with col2:
                # Bouton de téléchargement
                filename = f"cdss_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="💾 Télécharger JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()
            
            # Résumé textuel
            st.markdown("####  Résumé textuel")
            
            summary = f"""
            ## Rapport d'analyse CDSS
            **Date:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
            **Temps d'analyse:** {results.get('processing_time', 0):.2f} secondes
            
            ### Symptômes détectés: {len(results.get('symptoms', []))}
            {chr(10).join(['• ' + s.get('name', '') for s in results.get('symptoms', [])])}
            
            ### Hypothèses diagnostiques:
            {chr(10).join([f'{i+1}. {h.get("diagnosis", "")} (score: {h.get("score", 0):.2f})' 
                          for i, h in enumerate(results.get('hypotheses', []))])}
            
            ### Niveau de confiance: {results.get('confidence_level', 0):.2f}
            """
            
            st.text_area("Résumé:", summary, height=300)
            
            col3, col4 = st.columns([3, 1])
            with col4:
                st.download_button(
                    label="📥 Télécharger résumé",
                    data=summary,
                    file_name=f"cdss_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # ============================================================================
    # PIED DE PAGE
    # ============================================================================
    st.markdown("---")
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.caption("🏥 **Clinical Decision Support System**")
        st.caption("Version 1.0 • Multi-Agent AI")
    
    with col_footer2:
        st.caption(" **Avertissement médical**")
        st.caption("Cet outil est un support décisionnel. Consultez toujours un professionnel de santé.")
    
    with col_footer3:
        if st.session_state.get('last_analysis'):
            last_time = st.session_state.last_analysis.strftime('%H:%M:%S')
            st.caption(f"🕐 Dernière analyse: {last_time}")

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    # Initialisation des variables de session
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    
    # Exécuter l'application
    main()