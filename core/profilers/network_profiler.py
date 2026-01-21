"""
Network Profiler
================
Analiza densidad de red antes de ejecutar algoritmos de grafos.

Por qué es crítico:
- Los algoritmos de red (GNN, PageRank) son costosos O(n²)
- Si no hay densidad, no tiene sentido ejecutarlos
- Ahorro de costes de cómputo significativo
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class NetworkProfileStats:
    """Estadísticas del perfil de red."""
    nif: str
    num_counterparties: int
    total_edges: int
    avg_degree: float
    density: float
    is_dense_enough: bool
    recommended_algorithm: str  # 'NONE', 'SIMPLE', 'FULL_GRAPH'


class NetworkProfiler:
    """
    Perfila la densidad de red de transacciones.
    
    Uso:
        profiler = NetworkProfiler(min_edges=10, min_density=0.1)
        stats = profiler.profile(nif='B12345678', df_m347=...)
    """
    
    def __init__(
        self,
        min_edges: int = 10,
        min_density: float = 0.1,
        min_counterparties: int = 3
    ):
        """
        Args:
            min_edges: Mínimo de aristas para análisis de red
            min_density: Densidad mínima para análisis completo
            min_counterparties: Mínimo de contrapartes
        """
        self.min_edges = min_edges
        self.min_density = min_density
        self.min_counterparties = min_counterparties
    
    def profile(
        self,
        nif: str,
        df_m347: pd.DataFrame
    ) -> NetworkProfileStats:
        """
        Genera perfil de red para una empresa.
        
        Returns:
            NetworkProfileStats con métricas de red
        """
        if df_m347 is None or df_m347.empty:
            return NetworkProfileStats(
                nif=nif,
                num_counterparties=0,
                total_edges=0,
                avg_degree=0.0,
                density=0.0,
                is_dense_enough=False,
                recommended_algorithm='NONE'
            )
        
        # Filtrar operaciones de esta empresa
        company_ops = df_m347[
            (df_m347['nif_declarante'] == nif) |
            (df_m347['nif_contraparte'] == nif)
        ]
        
        if company_ops.empty:
            return NetworkProfileStats(
                nif=nif,
                num_counterparties=0,
                total_edges=0,
                avg_degree=0.0,
                density=0.0,
                is_dense_enough=False,
                recommended_algorithm='NONE'
            )
        
        # Contar contrapartes únicas
        counterparties = set(company_ops['nif_declarante'].unique()) | \
                        set(company_ops['nif_contraparte'].unique())
        counterparties.discard(nif)  # Remover el propio NIF
        
        num_counterparties = len(counterparties)
        total_edges = len(company_ops)
        
        # Calcular grado promedio (aproximado)
        avg_degree = total_edges / max(num_counterparties, 1)
        
        # Calcular densidad (edges / possible_edges)
        # Para grafo dirigido: density = edges / (n * (n-1))
        n = num_counterparties + 1  # +1 por la empresa central
        max_edges = n * (n - 1) if n > 1 else 1
        density = total_edges / max_edges
        
        # Determinar si es suficientemente densa
        is_dense = (
            num_counterparties >= self.min_counterparties and
            total_edges >= self.min_edges
        )
        
        # Recomendar algoritmo
        if not is_dense:
            algorithm = 'NONE'
        elif density < self.min_density:
            algorithm = 'SIMPLE'  # Solo flags básicos
        else:
            algorithm = 'FULL_GRAPH'  # GNN, PageRank completo
        
        return NetworkProfileStats(
            nif=nif,
            num_counterparties=num_counterparties,
            total_edges=total_edges,
            avg_degree=avg_degree,
            density=density,
            is_dense_enough=is_dense,
            recommended_algorithm=algorithm
        )
    
    def profile_all(
        self,
        nifs: list,
        df_m347: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perfila múltiples empresas.
        
        Returns:
            DataFrame con perfil de red de cada empresa
        """
        records = []
        
        for nif in nifs:
            stats = self.profile(nif, df_m347)
            records.append({
                'nif': stats.nif,
                'num_counterparties': stats.num_counterparties,
                'total_edges': stats.total_edges,
                'avg_degree': stats.avg_degree,
                'density': stats.density,
                'is_dense_enough': stats.is_dense_enough,
                'recommended_algorithm': stats.recommended_algorithm
            })
        
        return pd.DataFrame(records)
