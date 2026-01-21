"""
Script de verificación de conectividad y carga de datos para FraudHunter OS v3.0.
Simula la carga de datos desde Supabase y valida la estructura Gold.
"""

import pandas as pd
import numpy as np
import sys
import os

# Mock Streamlit to avoid errors in headless script execution
class MockSt:
    def cache_resource(self, **kwargs): return lambda f: f
    def cache_data(self, **kwargs): return lambda f: f
    def secrets(self): return {}
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def success(self, msg): print(f"SUCCESS: {msg}")

# Inject mock into sys.modules
import types
st_mock = MockSt()
mock_module = types.ModuleType('streamlit')
mock_module.cache_resource = st_mock.cache_resource
mock_module.cache_data = st_mock.cache_data
mock_module.secrets = {}
mock_module.session_state = {}
mock_module.error = st_mock.error
mock_module.warning = st_mock.warning
mock_module.info = st_mock.info
mock_module.success = st_mock.success
sys.modules['streamlit'] = mock_module

def verify_deployment():
    print("--- 🔍 FraudHunter OS v3.0 Deployment Verification ---")
    
    # 1. Test Supabase Connector Imports
    try:
        from supabase_connector import is_supabase_available, get_data_source_stats
        print("[OK] Supabase Connector imported successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to import supabase_connector: {e}")
        return

    # 2. Test Core Loaders
    try:
        from core.loaders.gold_dataset_loader import GoldDatasetLoader
        loader = GoldDatasetLoader()
        df = loader.load()
        print(f"[OK] GoldDatasetLoader loaded {len(df)} records.")
        
        # Validate Required Columns for Cockpit
        required = ['nif', 'final_score', 'benford_kl_divergence', 'tax_score']
        missing = [c for c in required if c not in df.columns]
        if not missing:
            print("[OK] All required columns for Cockpit are present.")
        else:
            print(f"[FAIL] Missing columns: {missing}")
            
    except Exception as e:
        print(f"[FAIL] GoldDatasetLoader error: {e}")

    # 3. Test Benford Analyzer
    try:
        from core.features.benford_analyzer import BenfordAnalyzer
        analyzer = BenfordAnalyzer()
        # Mock transaction data for a single company
        mock_data = pd.Series(np.random.uniform(100, 10000, 1000))
        result = analyzer.analyze(mock_data)
        print(f"[OK] BenfordAnalyzer calculated KL Divergence: {result.kl_divergence:.4f}")
    except Exception as e:
        print(f"[FAIL] BenfordAnalyzer error: {e}")

    print("\n--- ✅ Verification Complete ---")

if __name__ == "__main__":
    verify_deployment()
