---
description: Cómo restaurar una versión anterior del código usando Git cuando algo se rompe
---

# Restaurar Versión Anterior con Git

## Problema
Algo se rompió y necesitas volver a una versión funcional del código.

## Prerrequisitos
- GitHub Desktop instalado (contiene git.exe)
- Repositorio Git inicializado en el proyecto

## Pasos

### 1. Ver historial de commits
```powershell
# Usando git de GitHub Desktop
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" log --oneline -20
```

O revisar el archivo `.git/logs/HEAD` para ver timestamps exactos.

### 2. Identificar el commit funcional
- Los timestamps en `.git/logs/HEAD` están en formato Unix
- Buscar el commit cercano a la hora en que funcionaba
- Formato: `[hash_anterior] [hash_nuevo] [autor] [timestamp] [mensaje]`

### 3. Volver al branch main (si estás en detached HEAD)
```powershell
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" checkout main
```

### 4. Restaurar archivo específico de un commit anterior
```powershell
# Reemplazar COMMIT_HASH con el hash del commit funcional
# Reemplazar ARCHIVO con el nombre del archivo (ej: streamlit_app.py)
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" checkout COMMIT_HASH -- ARCHIVO
```

### 5. Verificar en GitHub Desktop
- Abrir GitHub Desktop
- Ir a pestaña "Changes"
- El archivo restaurado aparecerá con cambios

### 6. Probar que funciona
```powershell
streamlit run streamlit_app.py
```

### 7. Hacer commit de la restauración
- En GitHub Desktop: escribir mensaje descriptivo
- Click en "Commit to main"

### 8. Crear tag de versión estable (RECOMENDADO)
```powershell
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" tag -a v1.0-estable -m "Version estable"
```

### 9. Push a GitHub (backup en la nube)
```powershell
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" push origin main --tags
```

## Recuperar versión tagueada en el futuro
```powershell
# Volver a una versión específica tagueada
& "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" checkout v1.0-estable
```

## Tips
- **Siempre crear tags** antes de presentaciones importantes
- **Push frecuente** a GitHub para tener backup
- **Mensajes de commit descriptivos** para identificar fácilmente versiones funcionales
