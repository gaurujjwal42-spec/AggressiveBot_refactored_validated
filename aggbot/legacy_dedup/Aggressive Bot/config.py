# -*- coding: utf-8 -*-
"""
Configuration loader for the trading bot.

This module loads settings from 'config.json' and securely overrides
sensitive values (secrets) with environment variables.
"""

import json
import os
import sys
from dotenv import load_dotenv

# Determine the project root and load the .env file if it exists
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Load base configuration from the JSON file
try:
    with open(os.path.join(project_root, 'config.json')) as f:
        _config_data = json.load(f)
except FileNotFoundError:
    print("FATAL: config.json not found. Please create it from config.sample.json.")
    sys.exit(1)

# --- Secrets Management ---
# Fetch secrets from environment variables first, falling back to the JSON file.
# In production, environment variables should always be set, and secrets removed from config.json.
PRIVATE_KEY = os.getenv('PRIVATE_KEY') or _config_data.get('PRIVATE_KEY')
API_KEY = os.getenv('API_KEY') or _config_data.get('API_KEY')
API_SECRET = os.getenv('API_SECRET') or _config_data.get('API_SECRET_KEY') or _config_data.get('API_SECRET')

# --- General Configuration ---
# Dynamically set all other config values as module-level variables.
for key, value in _config_data.items():
    # Use globals() to set module-level variables.
    # Do not overwrite secrets that have already been loaded.
    if key.upper() not in globals():
        globals()[key.upper()] = value