@echo off
setlocal
cd /d "%~dp0"

REM (Optional) Use a venv named .venv next to this script
if not exist .venv (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

call .venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Starting trading bot in standalone mode...
py run_trading_bot.py

pause