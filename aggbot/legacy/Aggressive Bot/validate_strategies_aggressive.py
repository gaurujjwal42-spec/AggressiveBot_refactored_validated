import sys
import os
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add the parent directory to the sys.path to allow importing from main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming AdvancedPortfolioManager and TradingSignal are in main.py
try:
    from main import AdvancedPortfolioManager, TradingSignal
except ImportError as e:
    print(f"Error importing AdvancedPortfolioManager or TradingSignal from main.py: {e}")
    print("Please ensure main.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# Dummy config for AdvancedPortfolioManager
dummy_config = {
    'MAX_PORTFOLIO_RISK': 0.15,
    'TARGET_SHARPE': 1.5,
    'AVAILABLE_CAPITAL': 100000.0, # Example available capital
    'CAPITAL_ALLOCATION_PERCENTAGE': 0.05 # 5% of available capital per trade, matching default in main.py
}

# Instantiate AdvancedPortfolioManager
portfolio_manager = AdvancedPortfolioManager(dummy_config)

# Create a dummy TradingSignal
dummy_signal = TradingSignal(
    symbol="BTC/USD",
    signal_type="BUY",
    confidence=0.8,
    entry_price=30000.0,
    stop_loss=29000.0,
    take_profit=31000.0,
    position_size=0.0, # This will be calculated
    timeframe="4h",
    indicators={"RSI": 60, "MACD": "bullish"},
    sentiment={"overall": "positive"},
    risk_score=0.05,
    expected_return=0.03,
    max_drawdown=0.01,
    timestamp=datetime.now()
)

print("--- Validating calculate_optimal_position_size ---")
try:
    optimal_size = portfolio_manager.calculate_optimal_position_size(dummy_signal, dummy_signal.entry_price)
    print(f"Calculated optimal position size: {optimal_size}")

    # Basic assertion: check if the optimal size is positive
    assert optimal_size > 0, "Optimal position size should be positive"
    print("Assertion passed: Optimal position size is positive.")

    # You can add more specific assertions here based on expected values
    # For example, if you expect a certain range or a specific calculation result
    expected_size = (dummy_config['AVAILABLE_CAPITAL'] * dummy_config['CAPITAL_ALLOCATION_PERCENTAGE']) / dummy_signal.entry_price
    print(f"Expected position size based on dummy config: {expected_size}")
    assert abs(optimal_size - expected_size) < 0.0001, "Calculated position size does not match expected size"
    print("Assertion passed: Calculated position size matches expected size.")

    print("Validation successful for calculate_optimal_position_size.")

except Exception as e:
    print(f"Validation failed for calculate_optimal_position_size: {e}")

print("--- Validation complete ---")