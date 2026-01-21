"""
Governance Metrics
==================
Métricas de auditoría y control para Dashboard Tier-1.

Incluye:
- ROC/AUC calculation
- Precision-Recall curves
- Confusion Matrix monetizada
- PSI (Population Stability Index)
- Feature Drift detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)


@dataclass
class ModelPerformance:
    """Métricas de performance del modelo."""
    auc_roc: float
    auc_pr: float
    precision_at_k: Dict[int, float]  # {50: 0.8, 100: 0.75, ...}
    recall_at_k: Dict[int, float]
    optimal_threshold: float
    
    # ROC curve data
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds_roc: np.ndarray
    
    # PR curve data
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'optimal_threshold': self.optimal_threshold
        }


@dataclass
class MonetizedConfusionMatrix:
    """Matriz de confusión con impacto monetario."""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Impacto monetario
    fraud_caught_amount: float        # € Fraude detectado
    fraud_missed_amount: float        # € Fraude no detectado
    false_alarm_cost: float           # € Coste operativo de FP
    net_benefit: float                # € Beneficio neto
    
    # Ratios
    precision: float
    recall: float
    f1_score: float
    
    def to_dict(self) -> Dict:
        return {
            'tp': self.true_positives,
            'fp': self.false_positives,
            'tn': self.true_negatives,
            'fn': self.false_negatives,
            'fraud_caught_eur': self.fraud_caught_amount,
            'fraud_missed_eur': self.fraud_missed_amount,
            'false_alarm_cost_eur': self.false_alarm_cost,
            'net_benefit_eur': self.net_benefit,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


class GovernanceMetrics:
    """
    Calculador de métricas de governance para modelos de fraude.
    
    Uso:
        metrics = GovernanceMetrics()
        perf = metrics.calculate_performance(y_true, y_scores)
        cm = metrics.monetized_confusion_matrix(y_true, y_pred, amounts)
    """
    
    def __init__(
        self,
        cost_per_review: float = 50.0,  # € coste revisar un caso
        fraud_recovery_rate: float = 0.7  # % fraude recuperable
    ):
        """
        Args:
            cost_per_review: Coste operativo de revisar un caso (€)
            fraud_recovery_rate: Porcentaje de fraude recuperable si se detecta
        """
        self.cost_per_review = cost_per_review
        self.fraud_recovery_rate = fraud_recovery_rate
    
    def calculate_performance(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> ModelPerformance:
        """
        Calcula métricas completas de performance.
        
        Args:
            y_true: Etiquetas reales (0/1)
            y_scores: Scores predichos (0-1)
        
        Returns:
            ModelPerformance con todas las métricas
        """
        # ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        auc_roc = auc(fpr, tpr)
        
        # PR curve
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall_arr, precision_arr)
        
        # Optimal threshold (máximo F1)
        optimal_threshold = self._find_optimal_threshold(y_true, y_scores)
        
        # Precision@K y Recall@K
        precision_at_k = {}
        recall_at_k = {}
        for k in [50, 100, 200, 500]:
            if k <= len(y_scores):
                p, r = self._precision_recall_at_k(y_true, y_scores, k)
                precision_at_k[k] = p
                recall_at_k[k] = r
        
        return ModelPerformance(
            auc_roc=float(auc_roc),
            auc_pr=float(auc_pr),
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            optimal_threshold=optimal_threshold,
            fpr=fpr,
            tpr=tpr,
            thresholds_roc=thresholds_roc,
            precision_curve=precision_arr,
            recall_curve=recall_arr
        )
    
    def monetized_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
        threshold: float = 0.5
    ) -> MonetizedConfusionMatrix:
        """
        Calcula matriz de confusión con impacto monetario.
        
        Args:
            y_true: Etiquetas reales (0/1)
            y_pred: Predicciones binarias o scores
            amounts: Importes asociados a cada caso (€)
            threshold: Umbral para convertir scores a binario
        
        Returns:
            MonetizedConfusionMatrix con € y ratios
        """
        # Convertir a binario si son scores
        if y_pred.max() <= 1 and y_pred.min() >= 0 and len(np.unique(y_pred)) > 2:
            y_pred_binary = (y_pred >= threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Calcular montos
        # TP: Fraude detectado
        tp_mask = (y_true == 1) & (y_pred_binary == 1)
        fraud_caught = amounts[tp_mask].sum() * self.fraud_recovery_rate
        
        # FN: Fraude no detectado (pérdida)
        fn_mask = (y_true == 1) & (y_pred_binary == 0)
        fraud_missed = amounts[fn_mask].sum()
        
        # FP: Coste de revisar falsos positivos
        false_alarm_cost = fp * self.cost_per_review
        
        # Beneficio neto
        net_benefit = fraud_caught - fraud_missed - false_alarm_cost
        
        # Ratios
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return MonetizedConfusionMatrix(
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            fraud_caught_amount=float(fraud_caught),
            fraud_missed_amount=float(fraud_missed),
            false_alarm_cost=float(false_alarm_cost),
            net_benefit=float(net_benefit),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1)
        )
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calcula Population Stability Index (PSI).
        
        PSI mide cuánto ha cambiado la distribución de scores.
        
        Interpretación:
        - PSI < 0.10: Sin cambio significativo
        - PSI 0.10-0.25: Cambio moderado (investigar)
        - PSI > 0.25: Cambio significativo (reentrenar)
        
        Args:
            expected: Distribución de referencia (training)
            actual: Distribución actual (production)
            n_bins: Número de bins para discretizar
        
        Returns:
            PSI value
        """
        # Crear bins basados en expected
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Contar en cada bin
        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]
        
        # Convertir a proporciones
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Evitar división por cero
        epsilon = 1e-6
        expected_pct = np.clip(expected_pct, epsilon, 1)
        actual_pct = np.clip(actual_pct, epsilon, 1)
        
        # Calcular PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
    
    def calculate_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Calcula drift de features usando PSI.
        
        Returns:
            DataFrame con PSI por feature
        """
        drift_results = []
        
        for feature in features:
            if feature in reference_df.columns and feature in current_df.columns:
                ref_values = reference_df[feature].dropna().values
                cur_values = current_df[feature].dropna().values
                
                if len(ref_values) > 10 and len(cur_values) > 10:
                    psi = self.calculate_psi(ref_values, cur_values)
                    
                    drift_results.append({
                        'feature': feature,
                        'psi': psi,
                        'status': 'OK' if psi < 0.1 else ('WARNING' if psi < 0.25 else 'CRITICAL')
                    })
        
        return pd.DataFrame(drift_results)
    
    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """Encuentra el umbral que maximiza F1."""
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_scores >= threshold).astype(int)
            
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return float(best_threshold)
    
    def _precision_recall_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """Calcula Precision@K y Recall@K."""
        # Ordenar por score descendente
        sorted_indices = np.argsort(y_scores)[::-1]
        top_k_indices = sorted_indices[:k]
        
        # Precision@K = TP en top K / K
        tp_at_k = y_true[top_k_indices].sum()
        precision_k = tp_at_k / k
        
        # Recall@K = TP en top K / Total positivos
        total_positive = y_true.sum()
        recall_k = tp_at_k / total_positive if total_positive > 0 else 0
        
        return float(precision_k), float(recall_k)
