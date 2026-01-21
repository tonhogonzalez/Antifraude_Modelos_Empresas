import json
import os
from datetime import datetime, timedelta
import numpy as np

def generate_governance_data():
    data_dir = r"c:\2026\Apps_antigravity\Antifraude_Modelos_Empresas\model_governance\data"
    metrics_file = os.path.join(data_dir, "performance_metrics.json")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    history = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(31):
        date = base_date + timedelta(days=i)
        
        # Simular una ligera degradación y recuperación
        noise = np.random.normal(0, 0.01)
        auc = 0.92 + noise + (0.02 if i > 20 else 0)
        psi = 0.03 + abs(np.random.normal(0, 0.005)) + (0.05 if 10 < i < 15 else 0)
        precision = 0.85 + noise
        recall = 0.80 + noise
        
        entry = {
            "timestamp": date.isoformat(),
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": 2 * (precision * recall) / (precision + recall),
                "fpr": 0.12 - noise,
                "auc_roc": auc,
                "psi": psi,
                "drift_score": psi * 2  # Simpificamos drift proporcional al PSI
            }
        }
        history.append(entry)
        
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"Generated 31 days of governance data at {metrics_file}")

if __name__ == "__main__":
    generate_governance_data()
