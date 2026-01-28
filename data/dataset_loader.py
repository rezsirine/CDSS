"""
Chargement des datasets médicaux standards
SANS dépendance PyTorch obligatoire
"""

import pandas as pd
from typing import Dict, List, Tuple
import os
import json

# Import conditionnel de datasets (évite crash PyTorch)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception as e:
    print(f" HuggingFace datasets not available: {e}")
    print("   Using synthetic datasets only")
    DATASETS_AVAILABLE = False

class MedicalDatasetLoader:
    """
    Charge MIMIC-IV, MedQA, Symptom2Disease
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_medqa(self, split: str = "train", max_samples: int = 1000) -> pd.DataFrame:
        """
        Charge MedQA dataset
        """
        print(f" Loading MedQA ({split})...")
        
        if not DATASETS_AVAILABLE:
            print("   Using synthetic MedQA (datasets library unavailable)")
            return self._create_synthetic_medqa(max_samples)
        
        try:
            dataset = load_dataset(
                "bigbio/med_qa",
                name="med_qa_en_bigbio_qa",
                split=split,
                cache_dir=self.cache_dir
            )
            
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "source": "MedQA"
                })
            
            df = pd.DataFrame(data)
            print(f"    Loaded {len(df)} MedQA samples")
            return df
            
        except Exception as e:
            print(f"    MedQA loading failed: {e}")
            print("   Creating synthetic MedQA samples...")
            return self._create_synthetic_medqa(max_samples)
    
    def load_pubmedqa(self, split: str = "train", max_samples: int = 500) -> pd.DataFrame:
        """Charge PubMedQA"""
        print(f" Loading PubMedQA ({split})...")
        
        if not DATASETS_AVAILABLE:
            return pd.DataFrame()
        
        try:
            dataset = load_dataset(
                "pubmed_qa",
                "pqa_labeled",
                split=split,
                cache_dir=self.cache_dir
            )
            
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "context": " ".join(item["context"]["contexts"]),
                    "answer": item["final_decision"],
                    "source": "PubMedQA"
                })
            
            df = pd.DataFrame(data)
            print(f"    Loaded {len(df)} PubMedQA samples")
            return df
            
        except Exception as e:
            print(f"    PubMedQA failed: {e}")
            return pd.DataFrame()
    
    def load_symptom2disease(self, file_path: str = None) -> pd.DataFrame:
        """Charge Symptom2Disease"""
        print(" Loading Symptom2Disease...")
        
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"    Loaded {len(df)} samples from {file_path}")
            return df
        
        print("   Creating synthetic Symptom2Disease dataset...")
        return self._create_synthetic_symptom2disease()
    
    def load_mimic_simplified(self, max_samples: int = 200) -> pd.DataFrame:
        """MIMIC-IV simplifié"""
        print(" Loading MIMIC-IV (simplified)...")
        print("   Creating MIMIC-inspired synthetic data...")
        return self._create_synthetic_mimic(max_samples)
    
    def _create_synthetic_medqa(self, n_samples: int = 100) -> pd.DataFrame:
        """Crée données synthétiques style MedQA"""
        
        questions = [
            {
                "question": "A 45-year-old male presents with fever, cough, and fatigue for 3 days. What is the most likely diagnosis?",
                "choices": ["Influenza", "Pneumonia", "COVID-19", "Common cold"],
                "answer": "Influenza"
            },
            {
                "question": "A 35-year-old female with severe headache, nausea, and photophobia. Diagnosis?",
                "choices": ["Migraine", "Tension headache", "Meningitis", "Brain tumor"],
                "answer": "Migraine"
            },
            {
                "question": "68-year-old with chest pain, shortness of breath, and productive cough. Most likely?",
                "choices": ["Pneumonia", "Myocardial infarction", "COPD exacerbation", "Pulmonary embolism"],
                "answer": "Pneumonia"
            },
            {
                "question": "Patient with fever, abdominal pain, nausea, and diarrhea for 12 hours. Diagnosis?",
                "choices": ["Gastroenteritis", "Appendicitis", "Food poisoning", "IBS"],
                "answer": "Gastroenteritis"
            },
            {
                "question": "25-year-old with sudden severe headache, neck stiffness, and fever. Urgent diagnosis?",
                "choices": ["Meningitis", "Migraine", "Tension headache", "Sinusitis"],
                "answer": "Meningitis"
            }
        ]
        
        data = []
        while len(data) < n_samples:
            for q in questions:
                if len(data) >= n_samples:
                    break
                data.append({**q, "source": "Synthetic MedQA"})
        
        return pd.DataFrame(data)
    
    def _create_synthetic_symptom2disease(self) -> pd.DataFrame:
        """Crée dataset symptômes -> maladies"""
        
        data = [
            {"symptoms": ["fever", "cough", "fatigue"], "disease": "Influenza", "confidence": 0.85},
            {"symptoms": ["fever", "cough", "chest_pain", "shortness_of_breath"], "disease": "Pneumonia", "confidence": 0.90},
            {"symptoms": ["headache", "nausea", "photophobia"], "disease": "Migraine", "confidence": 0.88},
            {"symptoms": ["fever", "cough", "loss_of_taste"], "disease": "COVID-19", "confidence": 0.82},
            {"symptoms": ["nausea", "vomiting", "diarrhea", "abdominal_pain"], "disease": "Gastroenteritis", "confidence": 0.80},
            {"symptoms": ["headache", "neck_stiffness", "fever"], "disease": "Meningitis", "confidence": 0.92},
            {"symptoms": ["chest_pain", "shortness_of_breath", "palpitations"], "disease": "Myocardial Infarction", "confidence": 0.87},
            {"symptoms": ["cough", "wheezing", "shortness_of_breath"], "disease": "Asthma", "confidence": 0.85},
            {"symptoms": ["fatigue", "weight_loss", "night_sweats"], "disease": "Tuberculosis", "confidence": 0.75},
            {"symptoms": ["fever", "rash", "joint_pain"], "disease": "Dengue Fever", "confidence": 0.78}
        ]
        
        for item in data:
            item['symptoms_str'] = ", ".join(item['symptoms'])
        
        df = pd.DataFrame(data)
        print(f"    Created {len(df)} synthetic symptom-disease pairs")
        
        return df
    
    def _create_synthetic_mimic(self, n_samples: int = 200) -> pd.DataFrame:
        """Crée données style MIMIC-IV"""
        
        import random
        
        diagnoses = [
            "Sepsis", "Pneumonia", "Heart Failure", "COPD", "Diabetes",
            "Hypertension", "Acute Kidney Injury", "Stroke", "MI", "UTI"
        ]
        
        data = []
        for i in range(n_samples):
            diagnosis = random.choice(diagnoses)
            
            symptom_map = {
                "Sepsis": ["fever", "hypotension", "tachycardia", "altered_mental_status"],
                "Pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain"],
                "Heart Failure": ["shortness_of_breath", "edema", "fatigue", "orthopnea"],
                "COPD": ["shortness_of_breath", "cough", "wheezing"],
                "Diabetes": ["polyuria", "polydipsia", "weight_loss"],
                "Hypertension": ["headache", "dizziness"],
                "Acute Kidney Injury": ["decreased_urine", "edema", "fatigue"],
                "Stroke": ["weakness", "speech_difficulty", "facial_droop"],
                "MI": ["chest_pain", "shortness_of_breath", "nausea"],
                "UTI": ["dysuria", "fever", "frequent_urination"]
            }
            
            symptoms = symptom_map.get(diagnosis, ["symptom1", "symptom2"])
            
            data.append({
                "patient_id": f"P{i:05d}",
                "diagnosis": diagnosis,
                "symptoms": symptoms,
                "age": random.randint(30, 85),
                "gender": random.choice(["M", "F"]),
                "severity": random.choice(["Mild", "Moderate", "Severe"]),
                "source": "Synthetic MIMIC"
            })
        
        df = pd.DataFrame(data)
        print(f"    Created {len(df)} synthetic MIMIC-style records")
        
        return df
    
    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Charge tous les datasets"""
        
        print("\n" + "="*60)
        print("LOADING ALL MEDICAL DATASETS")
        print("="*60)
        
        datasets = {}
        
        # MedQA
        datasets['medqa'] = self.load_medqa(split="train", max_samples=100)
        
        # Symptom2Disease
        datasets['symptom2disease'] = self.load_symptom2disease()
        
        # MIMIC-inspired
        datasets['mimic'] = self.load_mimic_simplified(max_samples=200)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        for name, df in datasets.items():
            print(f"  {name}: {len(df)} samples")
        print("="*60)
        
        return datasets


def prepare_evaluation_data(datasets: Dict[str, pd.DataFrame]) -> Tuple[List, List, List]:
    """
    Prépare données pour évaluation
    """
    inputs = []
    ground_truth = []
    metadata = []
    
    # Symptom2Disease
    if 'symptom2disease' in datasets:
        df = datasets['symptom2disease']
        for _, row in df.iterrows():
            inputs.append(row['symptoms_str'])
            ground_truth.append(row['disease'])
            metadata.append({
                'source': 'symptom2disease',
                'symptoms': row.get('symptoms', []),
                'confidence': row.get('confidence', 0.8)
            })
    
    # MedQA
    if 'medqa' in datasets and len(datasets['medqa']) > 0:
        df = datasets['medqa'].head(30)
        for _, row in df.iterrows():
            inputs.append(row['question'])
            ground_truth.append(row['answer'])
            metadata.append({
                'source': 'medqa',
                'question': row['question']
            })
    
    # MIMIC
    if 'mimic' in datasets and len(datasets['mimic']) > 0:
        df = datasets['mimic'].head(20)
        for _, row in df.iterrows():
            symptom_str = ", ".join(row['symptoms'])
            inputs.append(symptom_str)
            ground_truth.append(row['diagnosis'])
            metadata.append({
                'source': 'mimic',
                'symptoms': row['symptoms']
            })
    
    return inputs, ground_truth, metadata
