"""
Système d'évaluation complet selon critères du projet
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ClinicalEvaluator:
    """
    Évaluation selon 3 critères du projet:
    1. Diagnostic (Accuracy, F1)
    2. Explication (Consistance, Plausibilité)
    3. Fiabilité (Score de confiance, Calibration)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {
            'diagnostic': {},
            'explanation': {},
            'reliability': {}
        }
    
    def evaluate_diagnostic(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_proba: List[float] = None
    ) -> Dict:
        """
        CRITÈRE 1: Performance Diagnostique
        
        Métriques:
        - Accuracy
        - F1-score (macro/weighted)
        - Precision
        - Recall
        - Confusion Matrix
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CRITÈRE 1: DIAGNOSTIC PERFORMANCE")
            print("="*60)
        
        # Encoder labels
        unique_labels = sorted(list(set(y_true + y_pred)))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        y_true_idx = [label_to_idx[y] for y in y_true]
        y_pred_idx = [label_to_idx[y] for y in y_pred]
        
        # Calculer métriques
        metrics = {
            'accuracy': accuracy_score(y_true_idx, y_pred_idx),
            'f1_macro': f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0),
            'precision': precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
            'recall': recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        
        # Classification report
        report = classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=unique_labels,
            zero_division=0,
            output_dict=True
        )
        
        metrics['confusion_matrix'] = cm.tolist()
        metrics['classification_report'] = report
        metrics['unique_labels'] = unique_labels
        
        if self.verbose:
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
        
        self.results['diagnostic'] = metrics
        return metrics
    
    def evaluate_explanations(
        self,
        explanations: List[Dict],
        ground_truth_symptoms: List[List[str]]
    ) -> Dict:
        """
        CRITÈRE 2: Qualité des Explications
        
        Métriques:
        - Consistance: Variance des importances
        - Plausibilité: Overlap avec ground truth
        - Complétude: Couverture des symptômes
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CRITÈRE 2: EXPLANATION QUALITY")
            print("="*60)
        
        consistency_scores = []
        plausibility_scores = []
        completeness_scores = []
        
        for exp, gt_symptoms in zip(explanations, ground_truth_symptoms):
            if not exp or not gt_symptoms:
                continue
            
            # Consistance
            if 'symptom_importance' in exp:
                importance_values = list(exp['symptom_importance'].values())
                if len(importance_values) > 1:
                    variance = np.var(importance_values)
                    consistency = 1.0 / (1.0 + variance)
                    consistency_scores.append(consistency)
            
            # Plausibilité
            if 'symptom_importance' in exp:
                important_symptoms = [
                    s for s, score in exp['symptom_importance'].items()
                    if score > 0.5
                ]
                
                if gt_symptoms:
                    overlap = len(set(important_symptoms) & set(gt_symptoms))
                    plausibility = overlap / len(gt_symptoms)
                    plausibility_scores.append(plausibility)
            
            # Complétude
            if 'symptom_importance' in exp and gt_symptoms:
                covered = sum(
                    1 for s in gt_symptoms
                    if s in exp['symptom_importance']
                )
                completeness = covered / len(gt_symptoms)
                completeness_scores.append(completeness)
        
        metrics = {
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'avg_plausibility': np.mean(plausibility_scores) if plausibility_scores else 0.0,
            'avg_completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
            'n_explanations': len(explanations)
        }
        
        if self.verbose:
            print(f"  Consistency:   {metrics['avg_consistency']:.4f}")
            print(f"  Plausibility:  {metrics['avg_plausibility']:.4f}")
            print(f"  Completeness:  {metrics['avg_completeness']:.4f}")
        
        self.results['explanation'] = metrics
        return metrics
    
    def evaluate_reliability(
        self,
        y_true: List[int],
        y_proba: List[float],
        confidence_scores: List[float]
    ) -> Dict:
        """
        CRITÈRE 3: Fiabilité (Score de confiance, Calibration)
        
        Métriques:
        - ECE (Expected Calibration Error)
        - MCE (Maximum Calibration Error)
        - Brier Score
        - Confidence distribution
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CRITÈRE 3: RELIABILITY & CALIBRATION")
            print("="*60)
        
        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true,
                y_proba,
                n_bins=10,
                strategy='uniform'
            )
            
            # ECE
            ece = np.mean(np.abs(prob_true - prob_pred))
            
            # MCE
            mce = np.max(np.abs(prob_true - prob_pred))
            
        except:
            ece = 1.0
            mce = 1.0
            prob_true = []
            prob_pred = []
        
        # Brier score
        brier = np.mean((np.array(y_true) - np.array(y_proba)) ** 2)
        
        # Confidence distribution
        conf_mean = np.mean(confidence_scores) if confidence_scores else 0.0
        conf_std = np.std(confidence_scores) if confidence_scores else 0.0
        
        metrics = {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'calibration_curve': {
                'prob_true': prob_true.tolist() if len(prob_true) > 0 else [],
                'prob_pred': prob_pred.tolist() if len(prob_pred) > 0 else []
            },
            'confidence_mean': float(conf_mean),
            'confidence_std': float(conf_std)
        }
        
        if self.verbose:
            print(f"  ECE:           {ece:.4f}")
            print(f"  Brier Score:   {brier:.4f}")
            print(f"  Conf. Mean:    {conf_mean:.4f}")
        
        self.results['reliability'] = metrics
        return metrics
    
    def benchmark_vs_baselines(
        self,
        baseline_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare contre baselines (BERT, GPT-4, etc.)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("BENCHMARK VS BASELINES")
            print("="*60)
        
        comparison = []
        
        # Système multi-agents
        comparison.append({
            'Model': 'Multi-Agent (Ours)',
            'Accuracy': self.results['diagnostic'].get('accuracy', 0),
            'F1-Score': self.results['diagnostic'].get('f1_macro', 0),
            'Explainability': self.results['explanation'].get('avg_plausibility', 0),
            'ECE': self.results['reliability'].get('ece', 1.0)
        })
        
        # Baselines
        for name, results in baseline_results.items():
            comparison.append({
                'Model': name,
                'Accuracy': results.get('accuracy', 0),
                'F1-Score': results.get('f1_macro', 0),
                'Explainability': results.get('explainability', 0),
                'ECE': results.get('ece', 1.0)
            })
        
        df = pd.DataFrame(comparison)
        
        if self.verbose:
            print(df.to_string(index=False))
        
        self.results['benchmark'] = df
        return df
    
    def save_results(self, output_dir: str = "./evaluation_results"):
        """Sauvegarde tous les résultats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON
        with open(f"{output_dir}/results.json", 'w') as f:
            # Convertir DataFrame en dict pour JSON
            results_copy = self.results.copy()
            if 'benchmark' in results_copy:
                results_copy['benchmark'] = results_copy['benchmark'].to_dict('records')
            
            json.dump(results_copy, f, indent=2)
        
        # Rapport texte
        self._generate_text_report(f"{output_dir}/report.txt")
        
        # Visualisations
        self._plot_results(f"{output_dir}/plots.png")
        
        print(f"\n✅ Results saved to {output_dir}/")
    
    def _generate_text_report(self, filepath: str):
        """Génère rapport texte"""
        lines = []
        lines.append("="*70)
        lines.append("CLINICAL MULTI-AGENT SYSTEM - EVALUATION REPORT")
        lines.append("="*70)
        lines.append("")
        
        # Diagnostic
        lines.append("1. DIAGNOSTIC PERFORMANCE")
        lines.append("-"*70)
        for k, v in self.results['diagnostic'].items():
            if k not in ['confusion_matrix', 'classification_report', 'unique_labels']:
                lines.append(f"  {k.upper()}: {v:.4f}")
        lines.append("")
        
        # Explanation
        lines.append("2. EXPLANATION QUALITY")
        lines.append("-"*70)
        for k, v in self.results['explanation'].items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k.upper()}: {v:.4f}")
        lines.append("")
        
        # Reliability
        lines.append("3. RELIABILITY & CALIBRATION")
        lines.append("-"*70)
        for k, v in self.results['reliability'].items():
            if k not in ['calibration_curve']:
                lines.append(f"  {k.upper()}: {v:.4f}")
        lines.append("")
        
        # Benchmark
        if 'benchmark' in self.results:
            lines.append("4. BENCHMARK COMPARISON")
            lines.append("-"*70)
            lines.append(self.results['benchmark'].to_string(index=False))
        
        lines.append("")
        lines.append("="*70)
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
    
    def _plot_results(self, filepath: str):
        """Génère visualisations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Diagnostic metrics
        if self.results['diagnostic']:
            ax = axes[0, 0]
            metrics = {k: v for k, v in self.results['diagnostic'].items() 
                      if k in ['accuracy', 'f1_macro', 'precision', 'recall']}
            ax.bar(metrics.keys(), metrics.values())
            ax.set_title('Diagnostic Performance')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Explanation quality
        if self.results['explanation']:
            ax = axes[0, 1]
            metrics = {k: v for k, v in self.results['explanation'].items()
                      if k in ['avg_consistency', 'avg_plausibility', 'avg_completeness']}
            ax.bar(metrics.keys(), metrics.values())
            ax.set_title('Explanation Quality')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # 3. Calibration curve
        if 'calibration_curve' in self.results['reliability']:
            ax = axes[1, 0]
            calib = self.results['reliability']['calibration_curve']
            if calib['prob_true']:
                ax.plot(calib['prob_pred'], calib['prob_true'], 'o-', label='Model')
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('True Probability')
                ax.set_title('Calibration Curve')
                ax.legend()
        
        # 4. Benchmark
        if 'benchmark' in self.results:
            ax = axes[1, 1]
            df = self.results['benchmark']
            df_plot = df.set_index('Model')[['Accuracy', 'F1-Score']]
            df_plot.plot(kind='bar', ax=ax)
            ax.set_title('Benchmark Comparison')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()