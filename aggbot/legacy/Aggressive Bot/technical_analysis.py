#!/usr/bin/env python3
"""
Technical Analysis Module

This module provides functions to calculate various technical indicators from
k-line (candlestick) data. It is designed to be used by the trading bot's
strategy and analysis components.

Indicators included:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Average True Range (ATR)
- Stochastic Oscillator (%K, %D)
- On-Balance Volume (OBV)
- Chaikin Money Flow (CMF)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

# --- Constants ---
# Standard lookback periods for indicators. These can be tuned.
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
CMF_PERIOD = 20

def _preprocess_klines(klines: List[List[Any]]) -> Optional[pd.DataFrame]:
    """
    Converts k-line data into a pandas DataFrame and performs basic validation.
    """
    if not klines or len(klines) < max(RSI_PERIOD, MACD_SLOW, CMF_PERIOD):
        # Not enough data to calculate all indicators
        return None

    # Standard kline format: [timestamp, open, high, low, close, volume, ...]
    # Handle different data formats flexibly
    num_cols = len(klines[0]) if klines else 0
    
    if num_cols >= 6:
        # Use basic columns that we need
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Add extra columns if available
        extra_columns = [f'col_{i}' for i in range(6, num_cols)]
        all_columns = base_columns + extra_columns
        df = pd.DataFrame(klines, columns=all_columns[:num_cols])
    else:
        # Not enough columns for basic analysis
        return None
    
    # Convert necessary columns to numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    if df.empty:
        return None
        
    return df

def _calculate_rsi(df: pd.DataFrame) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_macd(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd - macd_signal
    return {'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist}

def _calculate_atr(df: pd.DataFrame) -> pd.Series:
    """Calculates the Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    return atr

def _calculate_stochastic_oscillator(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculates the Stochastic Oscillator (%K and %D)."""
    low_min = df['low'].rolling(window=STOCH_K_PERIOD).min()
    high_max = df['high'].rolling(window=STOCH_K_PERIOD).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=STOCH_D_PERIOD).mean()
    return {'stoch_k': stoch_k, 'stoch_d': stoch_d}

def _calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculates the On-Balance Volume (OBV)."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def _calculate_cmf(df: pd.DataFrame) -> pd.Series:
    """Calculates the Chaikin Money Flow (CMF)."""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0) # Handle cases where high == low
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=CMF_PERIOD).sum() / df['volume'].rolling(window=CMF_PERIOD).sum()
    return cmf

def get_technical_indicators(klines: List[List[Any]]) -> Optional[Dict[str, float]]:
    """
    Calculates a comprehensive set of technical indicators from k-line data.

    Args:
        klines: A list of lists, where each inner list represents a candlestick.
                Expected format: [timestamp, open, high, low, close, volume, ...]

    Returns:
        A dictionary containing the latest values for all calculated indicators,
        or None if the data is insufficient.
    """
    df = _preprocess_klines(klines)
    if df is None:
        return None

    try:
        # Calculate all indicators
        rsi = _calculate_rsi(df)
        macd_data = _calculate_macd(df)
        atr = _calculate_atr(df)
        stoch_data = _calculate_stochastic_oscillator(df)
        obv = _calculate_obv(df)
        cmf = _calculate_cmf(df)

        # Calculate volume ratio (current volume vs 20-period average)
        volume_sma_20 = df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1.0
        
        # Calculate rate of change (ROC) for momentum
        roc_period = 10
        if len(df) >= roc_period:
            current_close = df['close'].iloc[-1]
            past_close = df['close'].iloc[-roc_period]
            roc = ((current_close - past_close) / past_close) * 100 if past_close > 0 else 0
        else:
            roc = 0
        
        # Compile the results into a dictionary with the latest values
        indicators = {
            'rsi': rsi.iloc[-1],
            'macd': macd_data['macd'].iloc[-1],
            'macd_signal': macd_data['macd_signal'].iloc[-1],
            'macd_hist': macd_data['macd_hist'].iloc[-1],
            'atr': atr.iloc[-1],
            'stoch_k': stoch_data['stoch_k'].iloc[-1],
            'stoch_d': stoch_data['stoch_d'].iloc[-1],
            'obv': obv.iloc[-1],
            'cmf': cmf.iloc[-1],
            'volume_sma_20': volume_sma_20,
            'last_volume': current_volume,
            'volume_ratio': volume_ratio,
            'roc': roc,
        }

        return indicators

    except Exception:
        # In case of any calculation error, return None to be safe
        return None

def calculate_all_indicators(klines: List[List[Any]]) -> Optional[Dict[str, float]]:
    """Alias for get_technical_indicators for backward compatibility"""
    return get_technical_indicators(klines)
