@echo off
setlocal

REM Move to script directory (project root)
cd /d "%~dp0"

REM Activate venv
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [ERROR] .venv not found. Create venv first: python -m venv .venv
  exit /b 1
)

REM Install deps if needed (optional)
REM pip install -r requirements-min.txt
REM pip install -r requirements-api.txt

REM Build index (idempotent)
python ragkit.py --config config.yaml index

REM Start API server
python api.py --host 127.0.0.1 --port 8000

endlocal