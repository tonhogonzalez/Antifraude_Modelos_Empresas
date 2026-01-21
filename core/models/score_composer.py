"""
Score Composer
==============
Combina múltiples scores intermedios en un score final explicable.

Por qué este diseño:
- Modelos monolíticos son difíciles de explicar y debuggear
- Scores desacoplados permiten identificar la fuente del riesgo
- La penalización por incertidumbre evita falsos negativos por falta de datos
- Cada componente puede evolucionar independientemente
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from ..features.benford_analyzer import BenfordAnalyzer


@dataclass
class FinalScore:
    """Resultado del ScoreComposer."""
    value: float                      # Score final 0-1
    components: Dict[str, float]      # Scores intermedios
    top_reasons: List[str]            # Top N razones explicables
    uncertainty_penalty: float        # Penalización por falta de datos
    confidence: str                   # 'HIGH', 'MEDIUM', 'LOW'
    
    def to_dict(self) -> Dict:
        return {
            'final_score': self.value,
            'tax_score': self.components.get('tax'),
            'network_score': self.components.get('network'),
            'financing_score': self.components.get('financing'),
            'uncertainty_penalty': self.uncertainty_penalty,
            'top_reasons': self.top_reasons,
            'confidence': self.confidence
        }


class ScoreComposer:
    """
    Orquestador de scoring que combina 3 scores → 1 Final.
    
    Scores intermedios:
    1. Tax Score: Solvencia estructural (anomalía M200)
    2. Network Score: Riesgo de red (carrusel, concentración)
    3. Financing Score: Deterioro diario (si hay datos bancarios)
    
    Uso:
        composer = ScoreComposer()
        composer.fit(df_empresas)
        result = composer.score(nif='B12345678')
    """
    
    def __init__(
        self,
        tax_weight: float = 0.5,
        network_weight: float = 0.3,
        financing_weight: float = 0.2,
        uncertainty_penalty_rate: float = 0.1,
        min_density_for_network: float = 0.1
    ):
        """
        Args:
            tax_weight: Peso del Tax Score en el final
            network_weight: Peso del Network Score
            financing_weight: Peso del Financing Score
            uncertainty_penalty_rate: Penalización por cada score faltante
            min_density_for_network: Densidad mínima de red para calcular Network Score
        """
        self.weights = {
            'tax': tax_weight,
            'network': network_weight,
            'financing': financing_weight
        }
        self.uncertainty_penalty_rate = uncertainty_penalty_rate
        self.min_density = min_density_for_network
        
        # Modelos internos
        self.tax_model: Optional[IsolationForest] = None
        self.scaler: Optional[RobustScaler] = None
        
        # Features por score
        self.tax_features = [
            'interest_burden', 'leverage', 'cash_buffer',
            'margen_neto', 'rotacion_activos', 'ratio_endeudamiento',
            'benford_score'
        ]
        self.network_features = [
            'reciprocity_score', 'flag_carousel_risk', 
            'concentracion_top3', 'flag_balance_anomaly'
        ]
        
        # Analyzer de Benford
        self.benford_analyzer = BenfordAnalyzer()
        
        # Datos internos
        self._df: Optional[pd.DataFrame] = None
    
    def fit(self, df: pd.DataFrame) -> 'ScoreComposer':
        """
        Entrena los modelos de scoring.
        
        Args:
            df: DataFrame con empresas y sus features
        """
        self._df = df.copy()
        
        # 1. Entrenar Tax Score (Isolation Forest)
        self._fit_tax_model(df)
        
        return self
    
    def _fit_tax_model(self, df: pd.DataFrame) -> None:
        """Entrena el modelo de Tax Score."""
        # Filtrar features existentes
        available_features = [f for f in self.tax_features if f in df.columns]
        
        if len(available_features) < 3:
            return
        
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        # Scaler robusto (menos sensible a outliers)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        self.tax_model = IsolationForest(
            contamination=0.05,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.tax_model.fit(X_scaled)
    
    def score(self, nif: str) -> Optional[FinalScore]:
        """
        Calcula score final para una empresa.
        
        Args:
            nif: NIF de la empresa
        
        Returns:
            FinalScore con componentes y explicaciones
        """
        if self._df is None:
            raise ValueError("Debes llamar fit() primero")
        
        company = self._df[self._df['nif'] == nif]
        if company.empty:
            return None
        
        company_row = company.iloc[0]
        
        # 1. Tax Score
        tax_score = self._calculate_tax_score(company_row)
        
        # 2. Network Score (solo si hay densidad)
        network_score = self._calculate_network_score(company_row)
        
        # 3. Financing Score
        financing_score = self._calculate_financing_score(company_row)
        
        # 4. Calcular score final con penalización
        final_score, penalty = self._compose_final(
            tax_score, network_score, financing_score
        )
        
        # 5. Generar explicaciones
        reasons = self._generate_reasons(company_row, tax_score, network_score, financing_score)
        
        # 6. Determinar confianza
        confidence = self._determine_confidence(tax_score, network_score, financing_score)
        
        return FinalScore(
            value=final_score,
            components={
                'tax': tax_score,
                'network': network_score,
                'financing': financing_score
            },
            top_reasons=reasons[:5],
            uncertainty_penalty=penalty,
            confidence=confidence
        )
    
    def score_all(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calcula scores para todas las empresas.
        
        Returns:
            DataFrame con columnas de score añadidas
        """
        if df is None:
            df = self._df
        
        if df is None:
            raise ValueError("No hay datos para scoring")
        
        df = df.copy()
        
        # Tax Score vectorizado
        df['tax_score'] = self._calculate_tax_score_batch(df)
        
        # Network Score vectorizado
        df['network_score'] = self._calculate_network_score_batch(df)
        
        # Financing Score vectorizado
        df['financing_score'] = self._calculate_financing_score_batch(df)
        
        # Final Score
        df['uncertainty_penalty'] = (
            df['network_score'].isna().astype(float) * self.uncertainty_penalty_rate +
            df['financing_score'].isna().astype(float) * self.uncertainty_penalty_rate
        )
        
        # Promediar disponibles con pesos ajustados
        df['final_score'] = self._compose_final_batch(df)
        
        # Clasificar riesgo
        df['risk_level'] = pd.cut(
            df['final_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        return df
    
    def _calculate_tax_score(self, row: pd.Series) -> Optional[float]:
        """Calcula Tax Score para una fila."""
        if self.tax_model is None:
            return None
        
        available_features = [f for f in self.tax_features if f in row.index]
        X = row[available_features].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0)
        X_scaled = self.scaler.transform(X)
        
        # decision_function: más negativo = más anómalo
        raw_score = self.tax_model.decision_function(X_scaled)[0]
        
        # Normalizar a 0-1 (más alto = más riesgo)
        # decision_function típicamente en [-0.5, 0.5]
        normalized = 1 - (raw_score + 0.5)
        return float(np.clip(normalized, 0, 1))
    
    def _calculate_tax_score_batch(self, df: pd.DataFrame) -> pd.Series:
        """Calcula Tax Score para batch."""
        if self.tax_model is None:
            return pd.Series(index=df.index, dtype=float)
        
        available_features = [f for f in self.tax_features if f in df.columns]
        X = df[available_features].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        raw_scores = self.tax_model.decision_function(X_scaled)
        normalized = 1 - (raw_scores + 0.5)
        return pd.Series(np.clip(normalized, 0, 1), index=df.index)
    
    def _calculate_network_score(self, row: pd.Series) -> Optional[float]:
        """Calcula Network Score (solo si hay datos de red)."""
        # Verificar densidad
        if 'num_contrapartes' in row.index:
            if row['num_contrapartes'] < 3:
                return None  # Incertidumbre por poca densidad
        
        score = 0.0
        count = 0
        
        for feature in self.network_features:
            if feature in row.index and pd.notna(row[feature]):
                score += float(row[feature]) if isinstance(row[feature], (int, float, bool)) else 0
                count += 1
        
        if count == 0:
            return None
        
        return min(score / count, 1.0)
    
    def _calculate_network_score_batch(self, df: pd.DataFrame) -> pd.Series:
        """Calcula Network Score para batch."""
        result = pd.Series(index=df.index, dtype=float)
        
        available = [f for f in self.network_features if f in df.columns]
        if not available:
            return result
        
        network_data = df[available].fillna(0)
        result = network_data.mean(axis=1)
        result = result.clip(0, 1)
        
        # Marcar como NaN si poca densidad
        if 'num_contrapartes' in df.columns:
            result[df['num_contrapartes'] < 3] = np.nan
        
        return result
    
    def _calculate_financing_score(self, row: pd.Series) -> Optional[float]:
        """Calcula Financing Score (si hay datos bancarios)."""
        # Sin datos bancarios, retornar None (incertidumbre)
        if 'stage' not in row.index:
            return None
        
        score = 0.0
        
        # Stage weighting
        if row.get('stage', 1) >= 2:
            score += 0.3
        if row.get('stage', 1) == 3:
            score += 0.3
        
        # Days in irregularity
        if row.get('days_in_irr', 0) > 30:
            score += 0.2
        if row.get('days_in_irr', 0) > 90:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_financing_score_batch(self, df: pd.DataFrame) -> pd.Series:
        """Calcula Financing Score para batch."""
        result = pd.Series(0.0, index=df.index)
        
        if 'stage' not in df.columns:
            return pd.Series(index=df.index, dtype=float)  # NaN
        
        result += (df['stage'] >= 2).astype(float) * 0.3
        result += (df['stage'] == 3).astype(float) * 0.3
        
        if 'days_in_irr' in df.columns:
            result += (df['days_in_irr'] > 30).astype(float) * 0.2
            result += (df['days_in_irr'] > 90).astype(float) * 0.2
        
        return result.clip(0, 1)
    
    def _compose_final(
        self, 
        tax: Optional[float], 
        network: Optional[float], 
        financing: Optional[float]
    ) -> Tuple[float, float]:
        """
        Compone score final con penalización por incertidumbre.
        
        Returns:
            (final_score, uncertainty_penalty)
        """
        scores = []
        weights = []
        penalty = 0.0
        
        if tax is not None:
            scores.append(tax)
            weights.append(self.weights['tax'])
        else:
            penalty += self.uncertainty_penalty_rate
        
        if network is not None:
            scores.append(network)
            weights.append(self.weights['network'])
        else:
            penalty += self.uncertainty_penalty_rate
        
        if financing is not None:
            scores.append(financing)
            weights.append(self.weights['financing'])
        else:
            penalty += self.uncertainty_penalty_rate
        
        if not scores:
            return 0.5, penalty  # Máxima incertidumbre
        
        # Promedio ponderado + penalización
        total_weight = sum(weights)
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight
        final = min(weighted_avg + penalty, 1.0)
        
        return final, penalty
    
    def _compose_final_batch(self, df: pd.DataFrame) -> pd.Series:
        """Compone score final para batch."""
        # Inicializar
        final = pd.Series(0.0, index=df.index)
        total_weight = pd.Series(0.0, index=df.index)
        
        for score_col, weight in [('tax_score', self.weights['tax']),
                                   ('network_score', self.weights['network']),
                                   ('financing_score', self.weights['financing'])]:
            if score_col in df.columns:
                valid_mask = df[score_col].notna()
                final[valid_mask] += df.loc[valid_mask, score_col] * weight
                total_weight[valid_mask] += weight
        
        # Evitar división por 0
        total_weight = total_weight.replace(0, 1)
        final = final / total_weight
        
        # Añadir penalización
        final = final + df['uncertainty_penalty']
        
        return final.clip(0, 1)
    
    def _generate_reasons(
        self, 
        row: pd.Series,
        tax_score: Optional[float],
        network_score: Optional[float],
        financing_score: Optional[float]
    ) -> List[str]:
        """Genera lista de razones explicables."""
        reasons = []
        
        # Tax reasons
        if tax_score and tax_score > 0.5:
            if row.get('leverage', 0) > 0.7:
                reasons.append(f"Apalancamiento elevado ({row['leverage']:.1%})")
            if row.get('interest_burden', 0) > 0.1:
                reasons.append(f"Carga financiera alta ({row['interest_burden']:.1%} de ventas)")
            if row.get('cash_buffer', 0) < 1.0:
                reasons.append(f"Buffer de liquidez bajo ({row['cash_buffer']:.1f})")
            if row.get('margen_neto', 0) < 0:
                reasons.append(f"Margen neto negativo ({row['margen_neto']:.1%})")
            if row.get('flag_benford_anomaly', False) or row.get('benford_score', 0) > 0.5:
                reasons.append(f"Anomalía en distribución de datos (Benford KL: {row.get('benford_kl_divergence', 0):.3f})")
        
        # Network reasons
        if network_score and network_score > 0.5:
            if row.get('flag_carousel_risk', False):
                reasons.append("Patrón de carrusel detectado (M349)")
            if row.get('flag_reciprocidad_sospechosa', False):
                reasons.append("Operaciones recíprocas sospechosas (M347)")
            if row.get('concentracion_top3', 0) > 0.8:
                reasons.append(f"Alta concentración en top 3 clientes ({row['concentracion_top3']:.0%})")
        
        # Financing reasons
        if financing_score and financing_score > 0.5:
            if row.get('stage', 1) >= 2:
                reasons.append(f"Stage IFRS9 deteriorado (Stage {row['stage']})")
            if row.get('days_in_irr', 0) > 30:
                reasons.append(f"Días en irregularidad: {row['days_in_irr']}")
        
        # Si no hay razones específicas
        if not reasons:
            if tax_score and tax_score > 0.5:
                reasons.append("Perfil fiscal atípico")
            if network_score and network_score > 0.5:
                reasons.append("Patrón de red inusual")
        
        return reasons
    
    def _determine_confidence(
        self,
        tax_score: Optional[float],
        network_score: Optional[float],
        financing_score: Optional[float]
    ) -> str:
        """Determina nivel de confianza basado en datos disponibles."""
        available = sum([
            tax_score is not None,
            network_score is not None,
            financing_score is not None
        ])
        
        if available >= 3:
            return 'HIGH'
        elif available >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
