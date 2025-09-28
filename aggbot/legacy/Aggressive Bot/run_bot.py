# -*- coding: utf-8 -*-
"""
This script is a legacy launcher and is no longer the primary way to start the bot.
Please use 'run_trading_bot.py' or 'start_trading_bot.bat' instead.
"""

import sys

def main():
    print("[WARN] This script (run_bot.py) is a legacy launcher and is deprecated.")
    print("[INFO] The primary way to start the bot is by using 'run_trading_bot.py'.")
    print("\n[INFO] Please run the following command instead:")
    print("       python run_trading_bot.py")
    print("\n[INFO] Or use the provided batch file: start_trading_bot.bat")
    sys.exit(1)

main()