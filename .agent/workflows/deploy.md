---
description: Cómo desplegar la nueva versión FraudHunter OS v3.0
---

# 🚀 Despliegue de FraudHunter OS v3.0

Esta guía detalla los pasos para desplegar la nueva versión con la interfaz **Enterprise OS** y el motor de análisis de **Benford**.

## 1. Instalación de Dependencias
Asegúrate de tener todas las librerías necesarias, incluyendo las nuevas para el análisis de Benford y el motor Gold.

```powershell
pip install -r requirements.txt
```

## 2. Ejecución Local (Desarrollo)
Si ya tienes una instancia de Streamlit corriendo, simplemente refresca el navegador. Si no, ejecuta:

```powershell
python -m streamlit run streamlit_app.py
```

## 3. Despliegue en Servidor (Producción)
Para entornos Tier-1, se recomienda ejecutar Streamlit tras un proxy inverso (Nginx) o usar un gestor de procesos como `pm2` para asegurar que el OS esté siempre online.

// turbo
```powershell
# Ejemplo con nohup para mantenerlo en background si no usas pm2
nohup python -m streamlit run streamlit_app.py --server.port 8501 > app.log 2>&1 &
```

## 4. Verificación Post-Despliegue
Una vez desplegado, verifica los siguientes puntos:
1.  **Navegación:** Que los botones `COCKPIT` y `GOVERNANCE` funcionen correctamente.
2.  **Benford Score:** Selecciona una empresa en el Cockpit y verifica que el KPI `BENFORD KL` muestre valores coherentes.
3.  **Governance:** Accede a la vista Governance para validar que los indicadores de AUC-ROC y PSI estén activos.

> [!TIP]
> Si encuentras algún error de importación, verifica que la carpeta `core/` esté en el PATH de Python.
