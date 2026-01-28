"""
Hypothesis Agent avec RAG PubMed - Version Complète
"""

import os
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import Entrez
import requests
from xml.etree import ElementTree as ET

# Configuration  pour PubMed
Entrez.email = "sirinerezgui585@gmail.com"  

class PubMedRetriever:
    """Récupérateur PubMed """
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.cache = {}
    
    def search_pubmed(self, query: str) -> List[Dict]:
        """Recherche articles PubMed"""
        
        # Vérifier cache
        if query in self.cache: # Si déjà recherché avant
            return self.cache[query]# Retourne résultat du cache
        
        try:
            # Recherche IDs
            handle = Entrez.esearch(
                db="pubmed", # Base de données PubMed
                term=query,# Termes de recherche (ex: "fever AND cough")
                retmax=self.max_results,
                sort="relevance" # Tri par pertinence
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                return []
            
            # Récupérer abstracts
            handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)# Lit les données
            handle.close()
            
            articles = []
            for article in records['PubmedArticle']:
                try:
                    medline = article['MedlineCitation']
                    abstract_text = ""
                    
                    if 'Abstract' in medline['Article']:
                        abstract_parts = medline['Article']['Abstract']['AbstractText']
                        if isinstance(abstract_parts, list):
                            abstract_text = " ".join([str(part) for part in abstract_parts])
                        else:
                            abstract_text = str(abstract_parts)
                    
                    articles.append({
                        "pmid": str(medline['PMID']),
                        "title": str(medline['Article']['ArticleTitle']),
                        "abstract": abstract_text,
                        "year": medline['Article']['Journal']['JournalIssue'].get('PubDate', {}).get('Year', 'N/A')
                    })
                except Exception as e:
                    continue
            
            self.cache[query] = articles
            return articles
            
        except Exception as e:
            print(f" PubMed error: {e}")
            return []


class HypothesisAgentWithRAG:
    """
    Agent de génération d'hypothèses avec RAG
    Remplace hypothesis_agent.py
    """
    
    def __init__(self, use_gpu: bool = False, verbose: bool = True):
        self.verbose = verbose
        self.use_gpu = use_gpu# Utiliser GPU ou non
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        # PubMed Retriever
        self.pubmed = PubMedRetriever(max_results=5)
        
        # modèle de langage médical (BioGPT) pour proposer des diagnostics
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(# Charge le tokenizer
                "microsoft/BioGPT-Large",
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/BioGPT-Large",
                local_files_only=True
            )
            if self.device >= 0:
                self.model = self.model.to(self.device)
            
            if verbose:
                print(" BioGPT-Large loaded")
        except:
            self.model = None
            self.tokenizer = None
            if verbose:#Si échec,  utilise des règles simples (ex: fièvre + toux = pneumonie possible)
                print(" BioGPT not available, using rules")
        
        # Knowledge base
        self.medical_kb = self._init_medical_kb()
    
    def _init_medical_kb(self) -> Dict:
        """Base de connaissances médicales"""
        return {
            "fever": ["Influenza", "Pneumonia", "COVID-19", "Sepsis"],
            "cough": ["Pneumonia", "Bronchitis", "Influenza", "COVID-19"],
            "headache": ["Migraine", "Tension Headache", "Meningitis"],
            "chest_pain": ["Pneumonia", "Myocardial Infarction", "Pulmonary Embolism"],
            "nausea": ["Gastroenteritis", "Migraine", "Pregnancy"],
            "shortness_of_breath": ["Pneumonia", "Asthma", "Heart Failure"]
        }
    
    def generate_hypotheses(self, symptoms: List[str], top_k: int = 5) -> List[Dict]:
        """
        Génère hypothèses avec RAG-enhanced generation
        """
        if self.verbose:
            print(f"\n Generating RAG-enhanced hypotheses...")
            print(f"   Symptoms: {symptoms}")
        
        # 1. Récupérer littérature PubMed
        rag_context = self._retrieve_medical_literature(symptoms)
        
        # 2. Générer hypothèses avec BioGPT + RAG
        if self.model and self.tokenizer and rag_context:
            hypotheses = self._generate_with_biogpt_rag(symptoms, rag_context, top_k)
        else:
            # Fallback intelligent
            hypotheses = self._generate_with_rules(symptoms, top_k)
        
        # 3. Enrichir avec sources PubMed
        for hyp in hypotheses:
            hyp['evidence_source'] = 'PubMed + BioGPT' if rag_context else 'Clinical Rules'
            hyp['rag_enhanced'] = bool(rag_context)
        
        if self.verbose:
            print(f"    Generated {len(hypotheses)} hypotheses (RAG: {bool(rag_context)})")
        
        return hypotheses
    
    def _retrieve_medical_literature(self, symptoms: List[str]) -> str:
        """Récupère littérature pertinente de PubMed"""
        all_abstracts = []
        
        # Construire requête
        symptom_query = " AND ".join(symptoms[:3])  # Limiter à 3 symptômes( Fièvre AND Toux AND Fatigue)
        query = f"{symptom_query} AND (diagnosis OR differential diagnosis)"
        
        if self.verbose:
            print(f"    Searching PubMed: '{query}'")
         # Recherche PubMed
        articles = self.pubmed.search_pubmed(query)
        
        if articles:
            for article in articles[:3]:  # Top 3 articles
                all_abstracts.append(f"[{article['pmid']}] {article['abstract'][:300]}")
            
            if self.verbose:
                print(f"    Retrieved {len(articles)} PubMed articles")
            
            return "\n\n".join(all_abstracts)
        
        return ""
    
    def _generate_with_biogpt_rag(self, symptoms: List[str], context: str, top_k: int) -> List[Dict]:
        """Génération avec BioGPT enrichi par RAG"""
        
        symptom_text = ", ".join(symptoms)
        
        # Prompt avec contexte RAG
        prompt = f"""Medical Literature Context:
{context[:800]}

Based on this literature and the following patient symptoms: {symptom_text}

Generate {top_k} most likely differential diagnoses with brief explanations:
1."""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=400,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parser les hypothèses
            hypotheses = self._parse_biogpt_output(generated, top_k)
            
            return hypotheses
            
        except Exception as e:
            if self.verbose:
                print(f"    BioGPT generation failed: {e}")
            return self._generate_with_rules(symptoms, top_k)
    
    def _parse_biogpt_output(self, text: str, top_k: int) -> List[Dict]:
        """Parse la sortie de BioGPT"""
        hypotheses = []
        
        # Extraire les lignes numérotées
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Nettoyer
                line = line.lstrip('0123456789.-) ')
                
                if ':' in line:
                    diagnosis, explanation = line.split(':', 1)
                    diagnosis = diagnosis.strip()
                    explanation = explanation.strip()
                else:
                    diagnosis = line
                    explanation = "Generated by BioGPT with RAG context"
                
                if diagnosis and len(diagnosis) > 2:
                    hypotheses.append({
                        "diagnosis": diagnosis,
                        "confidence": 0.75 - (len(hypotheses) * 0.05),
                        "explanation": explanation
                    })
                
                if len(hypotheses) >= top_k:
                    break
        
        # Compléter si nécessaire
        if len(hypotheses) < top_k:
            # Parser autrement
            parts = text.split('.')
            for part in parts:
                if len(hypotheses) >= top_k:
                    break
                
                part = part.strip()
                if len(part) > 10 and len(part) < 100:
                    hypotheses.append({
                        "diagnosis": part,
                        "confidence": 0.6,
                        "explanation": "Extracted from BioGPT output"
                    })
        
        return hypotheses[:top_k]
    
    def _generate_with_rules(self, symptoms: List[str], top_k: int) -> List[Dict]:
        """Génération basée sur règles médicales"""
        disease_scores = {}
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            
            # Chercher dans knowledge base
            for key, diseases in self.medical_kb.items():
                if key in symptom_lower or symptom_lower in key:
                    for disease in diseases:
                        if disease not in disease_scores:
                            disease_scores[disease] = 0
                        disease_scores[disease] += 1
        
        # Trier par score
        sorted_diseases = sorted(
            disease_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        hypotheses = []
        for disease, score in sorted_diseases[:top_k]:
            confidence = min(0.5 + (score * 0.15), 0.85)
            
            hypotheses.append({
                "diagnosis": disease,
                "confidence": confidence,
                "explanation": f"Rule-based match based on {score} symptom(s)"
            })
        
        return hypotheses


# Export
__all__ = ['HypothesisAgentWithRAG']