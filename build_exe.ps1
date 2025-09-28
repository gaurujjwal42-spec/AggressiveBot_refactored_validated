$ErrorActionPreference = "Stop"
. .\.venv\Scripts\Activate.ps1
pip install .[dev]
pyinstaller -F -n AggressiveBot.exe -c -i NONE `
  --collect-all aggbot `
  --hidden-import aggbot `
  -p . `
  -y `
  aggbot\__main__.py
Write-Host "EXE in dist\AggressiveBot.exe"