"""
Benford's Law Analyzer
======================
Detecta datos fabricados usando la Ley de Benford.

Por qué funciona:
- Los importes reales siguen la distribución de Benford (primer dígito)
- Los importes inventados tienden a distribución uniforme o sesgada
- La divergencia KL mide qué tan "artificial" es la distribución

Uso en fraude:
- Facturas falsas con números redondos
- Empresas pantalla con ventas fabricadas
- Carruseles con importes inventados
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Distribución teórica de Benford para primer dígito (1-9)
BENFORD_DISTRIBUTION = np.array([
    np.log10(1 + 1/d) for d in range(1, 10)
])


@dataclass
class BenfordAnalysis:
    """Resultado del análisis de Benford."""
    kl_divergence: float      # Divergencia KL (0 = perfecto, >0.1 = sospechoso)
    chi2_statistic: float     # Estadístico Chi-cuadrado
    chi2_p_value: float       # P-valor del test Chi2
    observed_dist: np.ndarray # Distribución observada
    expected_dist: np.ndarray # Distribución esperada (Benford)
    n_samples: int            # Número de muestras analizadas
    is_suspicious: bool       # Flag de sospecha
    confidence: str           # 'HIGH', 'MEDIUM', 'LOW'
    
    def to_dict(self) -> Dict:
        return {
            'kl_divergence': self.kl_divergence,
            'chi2_statistic': self.chi2_statistic,
            'chi2_p_value': self.chi2_p_value,
            'n_samples': self.n_samples,
            'is_suspicious': self.is_suspicious,
            'confidence': self.confidence
        }


class BenfordAnalyzer:
    """
    Analiza si una serie de importes sigue la Ley de Benford.
    
    Uso:
        analyzer = BenfordAnalyzer()
        result = analyzer.analyze(df['importe'])
        
        if result.is_suspicious:
            print(f"Datos sospechosos! KL={result.kl_divergence:.4f}")
    """
    
    def __init__(
        self,
        kl_threshold: float = 0.05,
        chi2_alpha: float = 0.05,
        min_samples: int = 50
    ):
        """
        Args:
            kl_threshold: Umbral de KL Divergence para marcar sospechoso
            chi2_alpha: Nivel de significancia para test Chi2
            min_samples: Mínimo de muestras para análisis fiable
        """
        self.kl_threshold = kl_threshold
        self.chi2_alpha = chi2_alpha
        self.min_samples = min_samples
        self.benford_dist = BENFORD_DISTRIBUTION
    
    def analyze(self, amounts: pd.Series) -> BenfordAnalysis:
        """
        Analiza una serie de importes contra la distribución de Benford.
        
        Args:
            amounts: Serie de importes monetarios
        
        Returns:
            BenfordAnalysis con métricas y flags
        """
        # Limpiar datos
        amounts = amounts.dropna()
        amounts = amounts[amounts.abs() > 0]  # Excluir ceros
        
        if len(amounts) < self.min_samples:
            return self._insufficient_data_result(len(amounts))
        
        # Extraer primer dígito
        first_digits = self._extract_first_digits(amounts)
        
        if len(first_digits) < self.min_samples:
            return self._insufficient_data_result(len(first_digits))
        
        # Calcular distribución observada
        observed_counts = np.array([
            (first_digits == d).sum() for d in range(1, 10)
        ])
        observed_dist = observed_counts / observed_counts.sum()
        
        # Distribución esperada (Benford)
        expected_dist = self.benford_dist
        
        # Calcular KL Divergence
        # KL(P || Q) = sum(P * log(P/Q))
        # Añadir epsilon para evitar log(0)
        epsilon = 1e-10
        kl_divergence = np.sum(
            observed_dist * np.log((observed_dist + epsilon) / (expected_dist + epsilon))
        )
        
        # Test Chi-cuadrado
        expected_counts = expected_dist * len(first_digits)
        chi2_stat, chi2_p = stats.chisquare(observed_counts, expected_counts)
        
        # Determinar si es sospechoso
        is_suspicious = (
            kl_divergence > self.kl_threshold or
            chi2_p < self.chi2_alpha
        )
        
        # Nivel de confianza
        if len(amounts) >= 500:
            confidence = 'HIGH'
        elif len(amounts) >= 100:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return BenfordAnalysis(
            kl_divergence=float(kl_divergence),
            chi2_statistic=float(chi2_stat),
            chi2_p_value=float(chi2_p),
            observed_dist=observed_dist,
            expected_dist=expected_dist,
            n_samples=len(amounts),
            is_suspicious=is_suspicious,
            confidence=confidence
        )
    
    def analyze_by_company(
        self, 
        df: pd.DataFrame,
        nif_col: str = 'nif',
        amount_col: str = 'importe'
    ) -> pd.DataFrame:
        """
        Analiza Benford por empresa.
        
        Returns:
            DataFrame con métricas por NIF
        """
        results = []
        
        for nif in df[nif_col].unique():
            company_amounts = df[df[nif_col] == nif][amount_col]
            analysis = self.analyze(company_amounts)
            
            results.append({
                'nif': nif,
                'benford_kl_divergence': analysis.kl_divergence,
                'benford_chi2_p_value': analysis.chi2_p_value,
                'benford_n_samples': analysis.n_samples,
                'benford_is_suspicious': analysis.is_suspicious,
                'benford_confidence': analysis.confidence
            })
        
        return pd.DataFrame(results)
    
    def _extract_first_digits(self, amounts: pd.Series) -> np.ndarray:
        """
        Extrae el primer dígito significativo de cada importe.
        """
        # Valor absoluto y convertir a string
        abs_amounts = amounts.abs()
        
        # Normalizar a notación científica y extraer primer dígito
        first_digits = []
        for amt in abs_amounts:
            if amt > 0:
                # Obtener primer dígito significativo
                s = f'{amt:.10e}'  # Notación científica
                first_char = s[0]
                if first_char.isdigit() and first_char != '0':
                    first_digits.append(int(first_char))
        
        return np.array(first_digits)
    
    def _insufficient_data_result(self, n: int) -> BenfordAnalysis:
        """Retorna resultado para datos insuficientes."""
        return BenfordAnalysis(
            kl_divergence=0.0,
            chi2_statistic=0.0,
            chi2_p_value=1.0,
            observed_dist=np.zeros(9),
            expected_dist=self.benford_dist,
            n_samples=n,
            is_suspicious=False,
            confidence='LOW'
        )
    
    def get_benford_score(self, kl_divergence: float) -> float:
        """
        Convierte KL Divergence a score 0-1 (más alto = más sospechoso).
        
        Escala:
        - KL < 0.01: Normal (score ~0.1)
        - KL 0.01-0.05: Leve (score 0.2-0.4)
        - KL 0.05-0.10: Moderado (score 0.4-0.7)
        - KL > 0.10: Alto (score 0.7-1.0)
        """
        if kl_divergence <= 0:
            return 0.0
        
        # Función sigmoide escalada
        # score = 1 / (1 + exp(-k*(kl - threshold)))
        k = 50  # Pendiente
        threshold = 0.05
        
        score = 1 / (1 + np.exp(-k * (kl_divergence - threshold)))
        return float(np.clip(score, 0, 1))


def calculate_benford_features(df: pd.DataFrame, amount_col: str = 'importe') -> pd.DataFrame:
    """
    Función de conveniencia para añadir features de Benford a un DataFrame.
    
    Añade columnas:
    - benford_kl_divergence
    - benford_score (0-1)
    - flag_benford_anomaly
    """
    analyzer = BenfordAnalyzer()
    
    # Análisis global
    global_analysis = analyzer.analyze(df[amount_col])
    
    df = df.copy()
    df['benford_kl_divergence'] = global_analysis.kl_divergence
    df['benford_score'] = analyzer.get_benford_score(global_analysis.kl_divergence)
    df['flag_benford_anomaly'] = global_analysis.is_suspicious
    
    return df
