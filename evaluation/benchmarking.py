import pandas as pd
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

class Benchmark:
    """Benchmark contre modèles non-explicatifs"""
    
    def __init__(self):
        self.datasets = {
            "MIMIC-IV": self._load_mimic,
            "MedQA": self._load_medqa,
            "Symptom2Disease": self._load_symptom2disease
        }
        
    def run_comparison(self):
        """Comparaison avec baselines (BERT, GPT-4, etc.)"""
        results = {}
        
        for dataset_name, loader in self.datasets.items():
            data = loader()
            
            # Évaluation du système multi-agents
            multi_agent_results = self.evaluate_multi_agent(data)
            
            # Évaluation des baselines
            baseline_results = self.evaluate_baselines(data)
            
            results[dataset_name] = {
                "multi_agent": multi_agent_results,
                "baselines": baseline_results
            }
            
        return results