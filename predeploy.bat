@echo off
setlocal
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install .[dev]
ruff aggbot
black --check aggbot
pytest -q
pyinstaller -F -n AggressiveBot.exe aggbot\__main__.py
echo Artifacts in dist\
endlocal
