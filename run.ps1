$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  winget install -e --id Python.Python.3.11
}
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
New-Item -ItemType Directory -Force -Path logs | Out-Null
aggbot run --cycles 1