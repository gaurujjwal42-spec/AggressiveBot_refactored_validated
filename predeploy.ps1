$ErrorActionPreference = 'Stop'
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .[dev]
ruff aggbot
black --check aggbot
pytest -q
pyinstaller -F -n AggressiveBot.exe aggbot/__main__.py
Write-Host 'Artifacts in dist/'
