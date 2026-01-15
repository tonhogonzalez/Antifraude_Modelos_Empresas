"""
Script para convertir archivos CSV a formato Parquet
El formato Parquet es 3-5x más rápido de cargar que CSV
"""

import pandas as pd
from pathlib import Path
import time

def convert_csv_to_parquet():
    """Convierte los archivos CSV principales a Parquet."""
    base_path = Path(__file__).parent
    
    files_to_convert = [
        ('data_empresas.csv', 'data_empresas.parquet'),
        ('data_eav.csv', 'data_eav.parquet'),
        ('data_m347.csv', 'data_m347.parquet'),
    ]
    
    for csv_name, parquet_name in files_to_convert:
        csv_path = base_path / csv_name
        parquet_path = base_path / parquet_name
        
        if not csv_path.exists():
            print(f"⚠️ No encontrado: {csv_name}")
            continue
            
        print(f"📄 Convirtiendo {csv_name}...")
        start = time.time()
        
        # Leer CSV
        df = pd.read_csv(csv_path)
        
        # Guardar como Parquet con compresión
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        elapsed = time.time() - start
        csv_size = csv_path.stat().st_size / (1024 * 1024)
        parquet_size = parquet_path.stat().st_size / (1024 * 1024)
        reduction = (1 - parquet_size / csv_size) * 100
        
        print(f"   ✅ {csv_name}: {csv_size:.1f}MB → {parquet_size:.1f}MB ({reduction:.0f}% reducción)")
        print(f"   ⏱️ Tiempo: {elapsed:.2f}s")
        print()
    
    print("🎉 Conversión completada!")


if __name__ == "__main__":
    convert_csv_to_parquet()
