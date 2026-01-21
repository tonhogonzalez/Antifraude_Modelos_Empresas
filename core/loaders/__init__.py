"""
FraudHunter Enterprise OS - Loaders Module
===========================================
Cargadores específicos para cada modelo tributario.

Why: Separar loaders por modelo permite:
1. Validación específica por formato
2. Optimización de memoria (solo cargar lo necesario)
3. Testing independiente
"""

from .base_loader import BaseLoader
from .m200_wide_loader import M200WideLoader
from .m347_loader import M347Loader
from .m349_loader import M349Loader
from .gold_dataset_loader import GoldDatasetLoader

__all__ = [
    "BaseLoader",
    "M200WideLoader",
    "M347Loader",
    "M349Loader",
    "GoldDatasetLoader",
]
