@echo off
setlocal
chcp 65001 >nul

echo [start] local-rag-kit

REM move to script dir (so it works from anywhere)
cd /d "%~dp0"

REM 1) venv check
if not exist ".venv\Scripts\python.exe" (
  echo [error] .venv not found.
  echo [hint] run:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install --upgrade pip
  echo   pip install -r requirements-min.txt
  echo   pip install -r requirements-api.txt
  echo   pip install -r requirements-sbert.txt
  exit /b 1
)

REM 2) ensure indexes
if not exist "index_tfidf\meta.json" (
  echo [index] building TF-IDF index...
  .venv\Scripts\python ragkit.py --config config_tfidf.yaml index
  if errorlevel 1 exit /b 1
) else (
  echo [index] TF-IDF index exists.
)

if not exist "index_sbert\meta.json" (
  echo [index] building SBERT index...
  .venv\Scripts\python ragkit.py --config config_sbert.yaml index
  if errorlevel 1 exit /b 1
) else (
  echo [index] SBERT index exists.
)

REM 3) optional: show data change status (API will also expose /check-data)
echo [check] current data fingerprint (from CLI)
.venv\Scripts\python -c "from ragkit import load_config, compute_data_fingerprint; cfg=load_config('config.yaml'); exts=list(getattr(getattr(cfg,'loader',None),'allowed_exts',[]) or []); print(compute_data_fingerprint(cfg.data_dir, exts))"
echo.

REM 4) run server
echo [api] http://127.0.0.1:8000
echo [api] press Ctrl+C to stop
.venv\Scripts\python api_hybrid.py --host 127.0.0.1 --port 8000
endlocal
