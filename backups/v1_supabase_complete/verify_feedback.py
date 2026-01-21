"""Verificar el dataset de feedback generado."""
from continuous_learning import get_feedback_store

store = get_feedback_store()

print("=" * 50)
print("FEEDBACK DATASET SUMMARY")
print("=" * 50)

# Conteo
c = store.get_sample_count()
print(f"\nTotal registros: {c['total']}")
print(f"  - Falsos Positivos: {c['false_positives']}")
print(f"  - Fraudes confirmados: {c['confirmed_fraud']}")
print(f"  - Watchlist (zona gris): {c['watchlist']}")

# Razones de rechazo
print("\n--- CAUSAS DE RECHAZO (FPs) ---")
r = store.analyze_rejection_reasons(90)
for reason, stats in r.get('by_reason', {}).items():
    print(f"  {reason}: {stats['count']} ({stats['percentage']:.1f}%)")

# Sectores con sugerencia
print("\n--- SECTORES PARA AJUSTAR UMBRAL ---")
for sector, stats in r.get('by_sector', {}).items():
    if stats.get('suggest_relax_threshold'):
        print(f"  ⚠️ {sector}: {stats['sector_normal_pct']:.0f}% son 'SECTOR_NORMAL'")

# Tipologías
print("\n--- TIPOLOGÍAS DE FRAUDE ---")
t = store.get_fraud_typology_stats(90)
for typ, stats in t.get('by_typology', {}).items():
    print(f"  {typ}: {stats['count']} casos")

# Watchlist
w = store.get_watchlist_nifs()
print(f"\nNIFs en Watchlist: {len(w)}")

# Estado
is_ready, reason = store.is_ready_for_training()
print(f"\n{'✅' if is_ready else '⏳'} {reason}")
