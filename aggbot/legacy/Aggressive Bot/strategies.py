#!/usr/bin/env python3
"""
Centralized Trading Strategies
"""

import pandas as pd
import numpy as np

# --- Simple Strategies ---

def simple_momentum_strategy(data, lookback=20, threshold=0.02):
    """Simple momentum strategy"""
    signals = []
    for i in range(lookback, len(data)):
        current_price = data.iloc[i]['price']
        past_price = data.iloc[i-lookback]['price']
        momentum = (current_price - past_price) / past_price
        if momentum > threshold:
            signals.append('BUY')
        elif momentum < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def simple_mean_reversion_strategy(data, lookback=20, std_threshold=2):
    """Simple mean reversion strategy"""
    signals = []
    for i in range(lookback, len(data)):
        recent_prices = data.iloc[i-lookback:i]['price']
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        current_price = data.iloc[i]['price']
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        if z_score < -std_threshold:
            signals.append('BUY')
        elif z_score > std_threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def simple_breakout_strategy(data, lookback=20, breakout_threshold=0.03):
    """Simple breakout strategy"""
    signals = []
    for i in range(lookback, len(data)):
        recent_prices = data.iloc[i-lookback:i]['price']
        high = recent_prices.max()
        low = recent_prices.min()
        current_price = data.iloc[i]['price']
        if current_price > high * (1 + breakout_threshold):
            signals.append('BUY')
        elif current_price < low * (1 - breakout_threshold):
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def simple_scalping_strategy(data, short_ma=5, long_ma=20):
    """Simple scalping strategy using moving averages"""
    signals = []
    for i in range(long_ma, len(data)):
        short_avg = data.iloc[i-short_ma:i]['price'].mean()
        long_avg = data.iloc[i-long_ma:i]['price'].mean()
        if short_avg > long_avg * 1.001:
            signals.append('BUY')
        elif short_avg < long_avg * 0.999:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

# --- Enhanced/Optimized Strategies ---

def enhanced_momentum_strategy(data, short_ma=5, long_ma=20, rsi_threshold=30):
    """Enhanced momentum strategy with RSI filter"""
    signals = []
    for i in range(long_ma, len(data)):
        if pd.isna(data.iloc[i]['sma_5']) or pd.isna(data.iloc[i]['sma_20']) or pd.isna(data.iloc[i]['rsi']):
            signals.append('HOLD')
            continue
        short_avg = data.iloc[i]['sma_5']
        long_avg = data.iloc[i]['sma_20']
        rsi = data.iloc[i]['rsi']
        if short_avg > long_avg * 1.002 and rsi < 70:
            signals.append('BUY')
        elif short_avg < long_avg * 0.998 and rsi > 30:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def enhanced_mean_reversion_strategy(data, bb_period=20, rsi_oversold=30, rsi_overbought=70):
    """Enhanced mean reversion using Bollinger Bands and RSI"""
    signals = []
    for i in range(bb_period, len(data)):
        if (pd.isna(data.iloc[i]['bb_upper']) or pd.isna(data.iloc[i]['bb_lower']) or
            pd.isna(data.iloc[i]['rsi'])):
            signals.append('HOLD')
            continue
        current_price = data.iloc[i]['price']
        bb_upper = data.iloc[i]['bb_upper']
        bb_lower = data.iloc[i]['bb_lower']
        rsi = data.iloc[i]['rsi']
        if current_price <= bb_lower and rsi <= rsi_oversold:
            signals.append('BUY')
        elif current_price >= bb_upper and rsi >= rsi_overbought:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def enhanced_breakout_strategy(data, lookback=20, volume_threshold=1.5):
    """Enhanced breakout strategy with volume confirmation"""
    signals = []
    for i in range(lookback, len(data)):
        recent_data = data.iloc[i-lookback:i]
        current_price = data.iloc[i]['price']
        current_volume = data.iloc[i]['volume']
        avg_volume = recent_data['volume'].mean()
        high = recent_data['price'].max()
        low = recent_data['price'].min()
        price_range = high - low
        volume_confirmed = current_volume > avg_volume * volume_threshold
        if current_price > high + (price_range * 0.01) and volume_confirmed:
            signals.append('BUY')
        elif current_price < low - (price_range * 0.01) and volume_confirmed:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def enhanced_scalping_strategy(data, fast_ma=3, slow_ma=8, trend_ma=21):
    """Enhanced scalping with trend filter"""
    signals = []
    for i in range(trend_ma, len(data)):
        if i < trend_ma:
            signals.append('HOLD')
            continue
        fast_avg = data.iloc[i-fast_ma:i]['price'].mean()
        slow_avg = data.iloc[i-slow_ma:i]['price'].mean()
        trend_avg = data.iloc[i-trend_ma:i]['price'].mean()
        current_price = data.iloc[i]['price']
        uptrend = current_price > trend_avg
        downtrend = current_price < trend_avg
        if fast_avg > slow_avg * 1.0005 and uptrend:
            signals.append('BUY')
        elif fast_avg < slow_avg * 0.9995 and downtrend:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

# --- Production Strategies (assumed implementations) ---
# These are placeholders since strategies.py was not provided for production.
# I've used the 'enhanced' versions as a stand-in.

def production_momentum_strategy(data):
    return enhanced_momentum_strategy(data)

def production_mean_reversion_strategy(data):
    return enhanced_mean_reversion_strategy(data)

def production_breakout_strategy(data):
    return enhanced_breakout_strategy(data)

def production_scalping_strategy(data):
    return enhanced_scalping_strategy(data)

# --- Ultra-Optimized Strategies ---

def ultra_momentum_strategy(data):
    """Ultra-optimized momentum strategy with multiple confirmations"""
    signals = []
    for i in range(50, len(data)):
        if (pd.isna(data.iloc[i]['sma_5']) or pd.isna(data.iloc[i]['sma_20']) or
            pd.isna(data.iloc[i]['rsi_14']) or pd.isna(data.iloc[i]['macd'])):
            signals.append('HOLD')
            continue
        sma_5 = data.iloc[i]['sma_5']
        sma_20 = data.iloc[i]['sma_20']
        rsi_14 = data.iloc[i]['rsi_14']
        macd = data.iloc[i]['macd']
        macd_signal = data.iloc[i]['macd_signal']
        price = data.iloc[i]['price']
        bullish_momentum = (
            sma_5 > sma_20 * 1.003 and rsi_14 > 45 and rsi_14 < 75 and
            macd > macd_signal and price > sma_5
        )
        bearish_momentum = (
            sma_5 < sma_20 * 0.997 and rsi_14 < 55 and rsi_14 > 25 and
            macd < macd_signal and price < sma_5
        )
        if bullish_momentum:
            signals.append('BUY')
        elif bearish_momentum:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def ultra_mean_reversion_strategy(data):
    """Ultra-optimized mean reversion with dynamic thresholds"""
    signals = []
    for i in range(25, len(data)):
        if (pd.isna(data.iloc[i]['bb_upper_20']) or pd.isna(data.iloc[i]['bb_lower_20']) or
            pd.isna(data.iloc[i]['rsi_14']) or pd.isna(data.iloc[i]['volatility'])):
            signals.append('HOLD')
            continue
        price = data.iloc[i]['price']
        bb_upper = data.iloc[i]['bb_upper_20']
        bb_lower = data.iloc[i]['bb_lower_20']
        bb_middle = data.iloc[i]['bb_middle_20']
        rsi_14 = data.iloc[i]['rsi_14']
        volatility = data.iloc[i]['volatility']
        vol_factor = min(volatility / data['volatility'].mean(), 2.0)
        rsi_oversold = 25 + (vol_factor * 5)
        rsi_overbought = 75 - (vol_factor * 5)
        oversold = (
            price <= bb_lower * 1.002 and rsi_14 <= rsi_oversold and
            price < bb_middle * 0.98
        )
        overbought = (
            price >= bb_upper * 0.998 and rsi_14 >= rsi_overbought and
            price > bb_middle * 1.02
        )
        if oversold:
            signals.append('BUY')
        elif overbought:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def ultra_breakout_strategy(data):
    """Ultra-optimized breakout with volume and volatility filters"""
    signals = []
    for i in range(30, len(data)):
        lookback = 25
        recent_data = data.iloc[i-lookback:i]
        current_price = data.iloc[i]['price']
        current_volume = data.iloc[i]['volume']
        if len(recent_data) < lookback:
            signals.append('HOLD')
            continue
        high = recent_data['price'].max()
        low = recent_data['price'].min()
        price_range = high - low
        avg_volume = recent_data['volume'].mean()
        volatility = recent_data['price'].std()
        breakout_threshold = max(price_range * 0.008, volatility * 0.5)
        volume_confirmed = current_volume > avg_volume * 1.3
        upward_breakout = (
            current_price > high + breakout_threshold and
            volume_confirmed and
            current_price > recent_data['price'].iloc[-5:].mean() * 1.005
        )
        downward_breakout = (
            current_price < low - breakout_threshold and
            volume_confirmed and
            current_price < recent_data['price'].iloc[-5:].mean() * 0.995
        )
        if upward_breakout:
            signals.append('BUY')
        elif downward_breakout:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

def ultra_scalping_strategy(data):
    """Ultra-optimized scalping with micro-trend detection"""
    signals = []
    for i in range(26, len(data)):
        if i < 26:
            signals.append('HOLD')
            continue
        fast_avg = data.iloc[i-2:i]['price'].mean()
        medium_avg = data.iloc[i-5:i]['price'].mean()
        slow_avg = data.iloc[i-10:i]['price'].mean()
        trend_avg = data.iloc[i-20:i]['price'].mean()
        current_price = data.iloc[i]['price']
        recent_volume = data.iloc[i-3:i]['volume'].mean()
        avg_volume = data.iloc[i-20:i]['volume'].mean()
        micro_uptrend = (
            fast_avg > medium_avg * 1.0008 and
            medium_avg > slow_avg * 1.0005 and
            current_price > trend_avg and
            recent_volume > avg_volume * 1.1
        )
        micro_downtrend = (
            fast_avg < medium_avg * 0.9992 and
            medium_avg < slow_avg * 0.9995 and
            current_price < trend_avg and
            recent_volume > avg_volume * 1.1
        )
        if micro_uptrend:
            signals.append('BUY')
        elif micro_downtrend:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

# --- Final/Advanced Strategies (as classes) ---

class AdvancedMomentumStrategy:
    def __init__(self):
        self.name = "Advanced Momentum"
    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < 50:
                signals.append('hold')
                continue
            current = df.iloc[i]
            prev = df.iloc[i-1]
            macd_bullish = current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']
            rsi_momentum = 50 < current['rsi'] < 80
            price_above_ema = current['close'] > current['ema_12']
            volume_confirmation = current['volume_ratio'] > 1.2
            volatility_ok = current['volatility'] < 0.05
            if (macd_bullish and rsi_momentum and price_above_ema and
                volume_confirmation and volatility_ok):
                signals.append('buy')
            elif (current['macd'] < current['macd_signal'] or current['rsi'] > 80 or
                  current['close'] < current['ema_12']):
                signals.append('sell')
            else:
                signals.append('hold')
        return signals

class AdvancedMeanReversionStrategy:
    def __init__(self):
        self.name = "Advanced Mean Reversion"
    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < 50:
                signals.append('hold')
                continue
            current = df.iloc[i]
            oversold = current['rsi'] < 25
            below_bb_lower = current['close'] < current['bb_lower']
            high_bb_width = current['bb_width'] > 0.04
            volume_spike = current['volume_ratio'] > 1.5
            if oversold and below_bb_lower and high_bb_width and volume_spike:
                signals.append('buy')
            elif (current['rsi'] > 60 or current['close'] > current['bb_middle']):
                signals.append('sell')
            else:
                signals.append('hold')
        return signals

class AdvancedBreakoutStrategy:
    def __init__(self):
        self.name = "Advanced Breakout"
    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < 50:
                signals.append('hold')
                continue
            current = df.iloc[i]
            price_breakout = current['close'] > current['bb_upper']
            volume_breakout = current['volume_ratio'] > 2.0
            momentum_confirm = current['macd'] > current['macd_signal']
            rsi_not_overbought = current['rsi'] < 75
            if (price_breakout and volume_breakout and momentum_confirm and rsi_not_overbought):
                signals.append('buy')
            elif (current['close'] < current['bb_middle'] or current['rsi'] > 80):
                signals.append('sell')
            else:
                signals.append('hold')
        return signals

class AdvancedScalpingStrategy:
    def __init__(self):
        self.name = "Advanced Scalping"
    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < 20:
                signals.append('hold')
                continue
            current = df.iloc[i]
            prev = df.iloc[i-1]
            ema_cross = current['close'] > current['ema_12'] and prev['close'] <= prev['ema_12']
            macd_momentum = current['macd_histogram'] > prev['macd_histogram']
            rsi_range = 45 < current['rsi'] < 65
            low_volatility = current['volatility'] < 0.03
            if ema_cross and macd_momentum and rsi_range and low_volatility:
                signals.append('buy')
            elif (current['close'] < current['ema_12'] or
                  abs(current['close'] - prev['close']) / prev['close'] > 0.02):
                signals.append('sell')
            else:
                signals.append('hold')
        return signals