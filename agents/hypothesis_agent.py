import re
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

class HypothesisAgent:
    """
    Agent de génération d'hypothèses diagnostiques via RAG avec PubMed
    """
    def __init__(self, use_gpu: bool = False, verbose: bool = True):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1

        if verbose:
            print("Initializing Hypothesis Agent...")

        # Charger un modèle médical léger
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")
            if self.device >= 0:
                self.model = self.model.to(self.device)
            if verbose:
                print("BioGPT model loaded")
        except Exception as e:
            if verbose:
                print(f"BioGPT not available: {e}. Using fallback.")
            self.model = None
            self.tokenizer = None

        # Base de connaissances médicales
        self.medical_knowledge = self._load_medical_knowledge()

    def _load_medical_knowledge(self) -> Dict:
        """Charge une base de connaissances médicales basique"""
        return {
            "symptom_disease_map": {
                "fever": ["influenza", "pneumonia", "covid-19", "urinary_tract_infection"],
                "cough": ["influenza", "pneumonia", "bronchitis", "covid-19", "asthma"],
                "headache": ["migraine", "tension_headache", "meningitis", "hypertension"],
                "nausea": ["gastroenteritis", "food_poisoning", "migraine", "pregnancy"],
                "chest_pain": ["myocardial_infarction", "pneumonia", "pulmonary_embolism", "anxiety"],
                "shortness_of_breath": ["pneumonia", "asthma", "copd", "heart_failure"],
                "abdominal_pain": ["appendicitis", "cholecystitis", "diverticulitis", "ibs"],
                "fatigue": ["anemia", "hypothyroidism", "depression", "chronic_fatigue_syndrome"]
            },
            "disease_descriptions": {
                "influenza": "Viral respiratory infection causing fever, cough, and body aches",
                "pneumonia": "Infection of the lung tissue causing cough, fever, and difficulty breathing",
                "covid-19": "Viral infection causing respiratory symptoms and fever",
                "migraine": "Neurological condition causing severe headaches and nausea",
                "appendicitis": "Inflammation of the appendix requiring surgical intervention",
                "myocardial_infarction": "Heart attack caused by blocked coronary arteries"
            }
        }

    def search_pubmed(self, symptoms: List[str], max_results: int = 5) -> List[Dict]:
        """Recherche dans PubMed (simulée pour l'instant)"""
        # Simulation de résultats PubMed
        pubmed_results = []

        for symptom in symptoms:
            if symptom in self.medical_knowledge["symptom_disease_map"]:
                diseases = self.medical_knowledge["symptom_disease_map"][symptom][:max_results]
                for disease in diseases:
                    pubmed_results.append({
                        "title": f"Clinical presentation of {disease.replace('_', ' ')}",
                        "abstract": self.medical_knowledge["disease_descriptions"].get(
                            disease, f"Medical condition involving {symptom}"
                        ),
                        "pmid": f"simulated_{disease}",
                        "relevance_score": 0.8
                    })

        return pubmed_results

    def generate_hypotheses(self, symptoms: List[str], top_k: int = 5) -> List[Dict]:
        """
        Génère des hypothèses diagnostiques avec RAG
        """
        try:
            # 1. Recherche dans PubMed/bases de connaissances
            pubmed_results = self.search_pubmed(symptoms, max_results=10)

            # 2. Construction du prompt contextuel
            prompt = self._build_rag_prompt(symptoms, pubmed_results)

            # 3. Génération des hypothèses
            if self.model and self.tokenizer:
                hypotheses = self._generate_with_biogpt(prompt, top_k)
            else:
                hypotheses = self._generate_fallback(symptoms, pubmed_results, top_k)

            # 4. Calcul des scores de confiance
            scored_hypotheses = self._score_hypotheses(hypotheses, symptoms)

            return scored_hypotheses

        except Exception as e:
            print(f" Error in hypothesis generation: {e}")
            return self._generate_fallback(symptoms, [], top_k)

    def _build_rag_prompt(self, symptoms: List[str], context_docs: List[Dict]) -> str:
        """Construit un prompt RAG pour la génération d'hypothèses"""
        symptom_text = ", ".join(symptoms)

        context = "\n".join([
            f"- {doc.get('title', 'Medical study')}: {doc.get('abstract', 'Clinical findings')}"
            for doc in context_docs[:5]  # Limiter à 5 docs pour le contexte
        ])

        prompt = f"""Based on the following patient symptoms and relevant medical literature, generate the most likely diagnostic hypotheses.

Patient Symptoms: {symptom_text}

Relevant Medical Literature:
{context}

Please provide the top 5 most likely diagnoses with brief explanations. Format as:
1. Diagnosis: Brief explanation based on symptoms and literature.
2. Diagnosis: Brief explanation...

Diagnoses:"""

        return prompt

    def _generate_with_biogpt(self, prompt: str, top_k: int) -> List[str]:
        """Génération avec BioGPT"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,
                    num_return_sequences=top_k,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            hypotheses = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Extraire seulement la partie après "Diagnoses:"
                if "Diagnoses:" in text:
                    diagnosis_part = text.split("Diagnoses:")[-1].strip()
                    hypotheses.extend([h.strip() for h in diagnosis_part.split('\n') if h.strip()])

            return hypotheses[:top_k] if hypotheses else self._generate_fallback([], [], top_k)

        except Exception as e:
            print(f" BioGPT generation failed: {e}")
            return self._generate_fallback([], [], top_k)

    def _generate_fallback(self, symptoms: List[str], context_docs: List[Dict], top_k: int) -> List[str]:
        """Génération de fallback basée sur les règles"""
        fallback_hypotheses = []

        # Collecter toutes les maladies possibles des symptômes
        all_diseases = set()
        for symptom in symptoms:
            if symptom in self.medical_knowledge["symptom_disease_map"]:
                all_diseases.update(self.medical_knowledge["symptom_disease_map"][symptom])

        # Trier par fréquence d'apparition dans les symptômes
        disease_scores = {}
        for disease in all_diseases:
            score = 0
            for symptom in symptoms:
                if symptom in self.medical_knowledge["symptom_disease_map"]:
                    if disease in self.medical_knowledge["symptom_disease_map"][symptom]:
                        score += 1
            disease_scores[disease] = score

        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)

        for disease, score in sorted_diseases[:top_k]:
            fallback_hypotheses.append(f"{disease.replace('_', ' ').title()}: Common condition associated with reported symptoms")

        return fallback_hypotheses

    def _score_hypotheses(self, hypotheses: List[str], symptoms: List[str]) -> List[Dict]:
        """Calcule les scores de confiance pour chaque hypothèse"""
        scored_hypotheses = []

        for hypothesis in hypotheses:
            # Extraire le nom de la maladie
            disease_match = re.match(r"^([^:]+):", hypothesis)
            disease_name = disease_match.group(1).strip() if disease_match else hypothesis.split(":")[0].strip()

            # Calculer le score basé sur la correspondance des symptômes
            confidence = self._calculate_confidence(disease_name.lower().replace(' ', '_'), symptoms)

            scored_hypotheses.append({
                "diagnosis": disease_name,
                "explanation": hypothesis,
                "confidence": confidence,
                "symptoms_matched": len([s for s in symptoms if self._symptom_matches_disease(s, disease_name.lower().replace(' ', '_'))])
            })

        # Trier par confiance
        scored_hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
        return scored_hypotheses[:5]  # Top 5

# Dans la méthode _score_hypotheses, améliorez le calcul de confiance :
def _calculate_confidence(self, disease: str, symptoms: List[str]) -> float:
    """Calcule un score de confiance amélioré pour une maladie donnée"""
    if disease not in self.medical_knowledge["symptom_disease_map"]:
        return 0.5  # Augmenté de 0.3 à 0.5
    
    disease_symptoms = set(self.medical_knowledge["symptom_disease_map"][disease])
    reported_symptoms = set(symptoms)
    
    # Intersection des symptômes
    matched_symptoms = disease_symptoms.intersection(reported_symptoms)
    
    # Score basé sur la proportion
    if len(disease_symptoms) > 0:
        base_score = len(matched_symptoms) / len(disease_symptoms)
    else:
        base_score = 0.6  # Augmenté
    
    # Bonus pour les bonnes correspondances
    if len(matched_symptoms) >= 2:
        base_score = min(base_score + 0.2, 1.0)
    if len(matched_symptoms) >= 3:
        base_score = min(base_score + 0.1, 1.0)
    
    # Pénalité pour symptômes manquants importants
    if len(disease_symptoms) > 0 and len(matched_symptoms) < len(disease_symptoms) / 2:
        base_score = max(base_score - 0.1, 0.3)
    
    return min(base_score, 1.0)
    def _symptom_matches_disease(self, symptom: str, disease: str) -> bool:
        """Vérifie si un symptôme correspond à une maladie"""
        if disease in self.medical_knowledge["symptom_disease_map"]:
            return symptom in self.medical_knowledge["symptom_disease_map"][disease]
        return False
