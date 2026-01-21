"""
Base Loader - Abstract interface for data loaders
==================================================
Todos los loaders heredan de esta clase base.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from dataclasses import dataclass


@dataclass
class LoaderStats:
    """Estadísticas de carga para auditoría."""
    rows_loaded: int
    rows_valid: int
    rows_invalid: int
    load_time_seconds: float
    source: str
    columns_found: List[str]
    columns_missing: List[str]


class BaseLoader(ABC):
    """
    Clase base abstracta para todos los loaders.
    
    Por qué este patrón:
    - Consistencia en la interfaz de carga
    - Validación estandarizada
    - Estadísticas de auditoría uniformes
    """
    
    def __init__(self, supabase_client=None):
        """
        Args:
            supabase_client: Cliente de Supabase para carga desde cloud
        """
        self.client = supabase_client
        self._stats: Optional[LoaderStats] = None
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Columnas requeridas para este loader."""
        pass
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Nombre de la tabla en Supabase."""
        pass
    
    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Carga datos desde la fuente especificada."""
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida y limpia los datos cargados."""
        pass
    
    def get_stats(self) -> Optional[LoaderStats]:
        """Retorna estadísticas de la última carga."""
        return self._stats
    
    def _validate_required_columns(self, df: pd.DataFrame) -> tuple:
        """
        Valida que existan las columnas requeridas.
        
        Returns:
            (columns_found, columns_missing)
        """
        found = [c for c in self.required_columns if c in df.columns]
        missing = [c for c in self.required_columns if c not in df.columns]
        return found, missing
