@echo off
setlocal
where python >nul 2>&1
if %errorlevel% neq 0 (
  echo Python not found. Installing via winget...
  winget install -e --id Python.Python.3.11
)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
if not exist logs mkdir logs
aggbot run --cycles 1
endlocal