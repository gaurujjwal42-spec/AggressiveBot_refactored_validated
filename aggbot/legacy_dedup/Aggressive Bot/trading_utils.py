#!/usr/bin/env python3
"""
Shared utility functions for the trading bot.
"""

import numpy as np
from typing import List

def calculate_volatility(prices: List[float], lookback: int = 20, annualized_factor: float = 24.0) -> float:
    """
    Calculate the annualized volatility of a series of prices.

    Args:
        prices: A list of recent prices.
        lookback: The number of recent prices to consider for the calculation.
        annualized_factor: The factor to annualize the volatility (e.g., 24 for hourly data to daily).

    Returns:
        The calculated annualized volatility, or a default value if not enough data.
    """
    if len(prices) < lookback:
        return 0.02  # Return a default volatility if not enough data

    price_changes = np.diff(prices) / prices[:-1]
    volatility = np.std(price_changes) * np.sqrt(annualized_factor)
    return float(volatility) if not np.isnan(volatility) else 0.02

if __name__ == "__main__":
    # This block will only run when the script is executed directly
    # It's a great way to test the functions in this file.
    
    # 1. Create a sample list of prices (e.g., hourly prices for a day)
    sample_prices = [
        100.0, 101.5, 102.0, 101.8, 103.2, 103.0, 104.5, 105.0, 104.8, 106.2,
        105.9, 107.3, 107.0, 106.5, 108.0, 108.2, 107.9, 109.5, 110.0, 109.8,
        108.5, 109.0, 110.2, 111.0
    ]
    
    print("--- Testing Volatility Calculation ---")
    
    # 2. Calculate volatility with default parameters
    volatility = calculate_volatility(sample_prices)
    print(f"Calculated Volatility (lookback=20, annualized_factor=24.0): {volatility:.4f}")
    
    # 3. Test with not enough data to show the default value is returned
    short_prices = [100.0, 101.0, 102.0]
    default_volatility = calculate_volatility(short_prices)
    print(f"Volatility with insufficient data (should return default): {default_volatility}")

    # 4. Test with different parameters
    volatility_custom = calculate_volatility(sample_prices, lookback=10, annualized_factor=365)
    print(f"Calculated Volatility (lookback=10, annualized_factor=365): {volatility_custom:.4f}")