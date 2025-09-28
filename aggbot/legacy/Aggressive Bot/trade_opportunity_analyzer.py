#!/usr/bin/env python3
"""
Advanced Trade Opportunity Analyzer
Contains sophisticated algorithms for analyzing trade opportunities with multi-timeframe analysis,
pattern recognition, market regime detection, and advanced scoring systems.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Handle technical analysis library imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

try:
    import logging

    logger = logging.getLogger(__name__)
    import joblib
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def analyze_fundamentals(
    pair: Dict[str, Any],
    open_symbols: set,
    optimized_params: Any,
    config: Dict,
    log_message: callable = None
) -> bool:
    """Performs fundamental and basic filtering. Returns True if the pair passes."""
    volume_24h = float(pair.get('volume', {}).get('h24', 0))
    change_24h = float(pair.get('priceChange24h', 0))
    market_cap = float(pair.get('marketCap', 0))
    liquidity_usd = float(pair.get('liquidity', {}).get('usd', 0))
    symbol = pair['baseToken']['symbol'].upper()
    base_trade_amount = config.get('TRADE_USDT', 10)

    if market_cap <= 0 or volume_24h <= 0:
        if log_message:
            log_message(f"❌ {symbol}: Invalid data - MC: ${market_cap:,.0f}, Vol: ${volume_24h:,.0f}", "DEBUG")
        return False

    is_not_already_owned = symbol not in open_symbols
    is_valid_market_cap = market_cap > config.get('MIN_MARKET_CAP', 500_000)
    
    dynamic_volume_threshold = config.get('MIN_24H_VOLUME', 1_000_000) * optimized_params.volume_threshold
    dynamic_momentum_min = config.get('MIN_24H_CHANGE', 5) * optimized_params.momentum_threshold * 100
    dynamic_momentum_max = config.get('MAX_24H_CHANGE', 50) * optimized_params.take_profit_multiplier
    
    # More aggressive volume threshold for higher trade frequency
    is_high_volume = volume_24h > dynamic_volume_threshold * 0.8
    
    # Reduced liquidity requirement for more trading opportunities
    is_sufficient_liquidity = liquidity_usd > base_trade_amount * 50
    
    # Wider momentum range to catch more opportunities    # Dynamic momentum range based on optimized parameters
    # The `momentum_threshold` from `optimized_params` is a factor, not a percentage.
    # We multiply it by 100 to convert it to a percentage for comparison with `change_24h`.
    # The `take_profit_multiplier` can also influence the upper bound of acceptable momentum.
    is_positive_momentum = dynamic_momentum_min * 0.6 < change_24h < dynamic_momentum_max * 1.2
    
    # Calculate volume to market cap ratio with more aggressive thresholds
    volume_to_mc_ratio = volume_24h / market_cap if market_cap > 0 else 0
    is_healthy_ratio = 0.05 < volume_to_mc_ratio < 8.0

    # Debug logging for rejections
    if log_message and not (is_not_already_owned and is_valid_market_cap and is_high_volume and
                           is_sufficient_liquidity and is_positive_momentum and is_healthy_ratio):
        rejection_reasons = []
        if not is_not_already_owned:
            rejection_reasons.append("Already owned")
        if not is_valid_market_cap:
            rejection_reasons.append(f"Low MC: ${market_cap:,.0f} < ${config.get('MIN_MARKET_CAP', 500_000):,.0f}")
        if not is_high_volume:
            rejection_reasons.append(f"Low Vol: ${volume_24h:,.0f} < ${dynamic_volume_threshold:,.0f}")
        if not is_sufficient_liquidity:
            rejection_reasons.append(f"Low Liq: ${liquidity_usd:,.0f} < ${base_trade_amount * 100:,.0f}")
        if not is_positive_momentum:
            rejection_reasons.append(f"Bad momentum: {change_24h:.2f}% not in [{dynamic_momentum_min:.2f}%, {dynamic_momentum_max:.2f}%]")
        if not is_healthy_ratio:
            rejection_reasons.append(f"Bad Vol/MC ratio: {volume_to_mc_ratio:.3f} not in [0.1, 5.0]")
        
        log_message(f"❌ {symbol}: {', '.join(rejection_reasons)}", "DEBUG")

    return (is_not_already_owned and is_valid_market_cap and is_high_volume and
            is_sufficient_liquidity and is_positive_momentum and is_healthy_ratio)

def _get_ml_bonus(
    symbol: str,
    last_indicator_data: pd.Series,
    ml_model: Any,
    ml_model_features: List[str],
    config: Dict,
    log_message: callable
) -> int:
    """Performs ML-based prediction and returns a confidence bonus."""
    if not config.get('USE_ML_MODEL', False) or ml_model is None:
        return 0
    try:
        model_input_df = pd.DataFrame([last_indicator_data])
        for col in ml_model_features:
            if col not in model_input_df.columns:
                model_input_df[col] = 0
        model_input_df = model_input_df[ml_model_features]

        prediction = ml_model.predict(model_input_df)[0]
        probability = ml_model.predict_proba(model_input_df)[0][1]

        if prediction == 1:
            ml_bonus = config.get('ML_CONFIDENCE_BONUS', 2)
            log_message(f"ML Model Signal for {symbol}: BUY (Prob: {probability:.2f}, Bonus: +{ml_bonus})", "INFO")
            return ml_bonus
    except Exception as e:
        log_message(f"Error during ML prediction for {symbol}: {e}", "WARNING")
    return 0

# Advanced Market Analysis Functions
def detect_market_regime(hist_data: pd.DataFrame, lookback_periods: int = 50) -> Dict[str, Any]:
    """Detect current market regime using multiple indicators and statistical analysis"""
    try:
        if hist_data is None or len(hist_data) < lookback_periods:
            return {'regime': 'unknown', 'confidence': 0, 'volatility_regime': 'normal'}
        
        # Calculate returns and volatility
        returns = hist_data['close'].pct_change().dropna()
        recent_returns = returns.tail(lookback_periods)
        
        # Volatility regime detection
        volatility = returns.rolling(20).std()
        current_vol = volatility.iloc[-1]
        vol_percentile = stats.percentileofscore(volatility.dropna(), current_vol)
        
        if vol_percentile > 80:
            vol_regime = 'high'
        elif vol_percentile < 20:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        # Trend regime detection using multiple timeframes
        short_ma = hist_data['close'].rolling(10).mean()
        medium_ma = hist_data['close'].rolling(20).mean()
        long_ma = hist_data['close'].rolling(50).mean()
        
        # Current trend alignment
        current_price = hist_data['close'].iloc[-1]
        trend_score = 0
        
        if current_price > short_ma.iloc[-1]:
            trend_score += 1
        if current_price > medium_ma.iloc[-1]:
            trend_score += 1
        if current_price > long_ma.iloc[-1]:
            trend_score += 1
        if short_ma.iloc[-1] > medium_ma.iloc[-1]:
            trend_score += 1
        if medium_ma.iloc[-1] > long_ma.iloc[-1]:
            trend_score += 1
        
        # Momentum analysis
        momentum = (current_price / hist_data['close'].iloc[-20] - 1) * 100
        
        # Market regime classification
        if trend_score >= 4 and momentum > 5:
            regime = 'strong_uptrend'
            confidence = min(95, 60 + trend_score * 7)
        elif trend_score >= 3 and momentum > 2:
            regime = 'uptrend'
            confidence = min(85, 50 + trend_score * 7)
        elif trend_score <= 1 and momentum < -5:
            regime = 'strong_downtrend'
            confidence = min(95, 60 + (5 - trend_score) * 7)
        elif trend_score <= 2 and momentum < -2:
            regime = 'downtrend'
            confidence = min(85, 50 + (5 - trend_score) * 7)
        else:
            regime = 'sideways'
            confidence = 40 + abs(momentum) * 2
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility_regime': vol_regime,
            'trend_score': trend_score,
            'momentum': momentum,
            'volatility_percentile': vol_percentile
        }
        
    except Exception as e:
        return {'regime': 'unknown', 'confidence': 0, 'volatility_regime': 'normal', 'error': str(e)}

def analyze_multi_timeframe_signals(symbol: str, get_historical_data_func: callable, 
                                  calculate_indicators_func: callable) -> Dict[str, Any]:
    """Analyze signals across multiple timeframes for comprehensive view"""
    try:
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        timeframe_scores = {}
        overall_signal = 0
        signal_strength = 0
        
        for tf in timeframes:
            try:
                hist_data = get_historical_data_func(f"{symbol}USDT", interval=tf, limit=100)
                if hist_data is None or len(hist_data) < 50:
                    continue
                
                indicators = calculate_indicators_func(hist_data)
                if indicators is None or indicators.empty:
                    continue
                
                last = indicators.iloc[-1]
                tf_score = 0
                
                # RSI analysis
                rsi = last.get('RSI_14', 50)
                if 30 < rsi < 70:
                    tf_score += 1
                elif rsi < 30:
                    tf_score += 2  # Oversold bonus
                
                # MACD analysis
                macd = last.get('MACD_12_26_9', 0)
                macd_signal = last.get('MACDs_12_26_9', 0)
                if macd > macd_signal:
                    tf_score += 1
                
                # EMA trend analysis
                ema_20 = last.get('EMA_20', 0)
                ema_50 = last.get('EMA_50', 0)
                if ema_20 > ema_50:
                    tf_score += 1
                
                # Volume analysis
                volume_ratio = last.get('volume', 0) / last.get('volume_sma_20', 1)
                if volume_ratio > 1.2:
                    tf_score += 1
                
                # Timeframe weight (higher timeframes get more weight)
                weight = {'5m': 1, '15m': 1.5, '1h': 2, '4h': 2.5, '1d': 3}.get(tf, 1)
                weighted_score = tf_score * weight
                
                timeframe_scores[tf] = {
                    'score': tf_score,
                    'weighted_score': weighted_score,
                    'weight': weight
                }
                
                overall_signal += weighted_score
                signal_strength += weight
                
            except Exception as e:
                continue
        
        # Normalize signal strength
        if signal_strength > 0:
            normalized_signal = overall_signal / signal_strength
        else:
            normalized_signal = 0
        
        return {
            'timeframe_scores': timeframe_scores,
            'overall_signal': overall_signal,
            'normalized_signal': normalized_signal,
            'signal_strength': signal_strength,
            'recommendation': 'BUY' if normalized_signal > 2.5 else 'HOLD' if normalized_signal > 1.5 else 'AVOID'
        }
        
    except Exception as e:
        return {'error': str(e), 'recommendation': 'AVOID'}

def detect_chart_patterns(hist_data: pd.DataFrame) -> Dict[str, Any]:
    """Detect various chart patterns using price action analysis"""
    try:
        if hist_data is None or len(hist_data) < 20:
            return {'patterns': [], 'pattern_score': 0}
        
        patterns_detected = []
        pattern_score = 0
        
        # Get recent price data
        highs = hist_data['high'].values
        lows = hist_data['low'].values
        closes = hist_data['close'].values
        volumes = hist_data['volume'].values
        
        # Double Bottom Pattern
        if len(closes) >= 20:
            recent_lows = lows[-20:]
            min_indices = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    min_indices.append(i)
            
            if len(min_indices) >= 2:
                last_two_mins = min_indices[-2:]
                if abs(recent_lows[last_two_mins[0]] - recent_lows[last_two_mins[1]]) / recent_lows[last_two_mins[0]] < 0.02:
                    patterns_detected.append('double_bottom')
                    pattern_score += 3
        
        # Ascending Triangle
        if len(closes) >= 15:
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Check for horizontal resistance
            resistance_level = np.mean(recent_highs[-5:])
            resistance_touches = sum(1 for h in recent_highs if abs(h - resistance_level) / resistance_level < 0.01)
            
            # Check for ascending support
            support_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if resistance_touches >= 2 and support_slope > 0:
                patterns_detected.append('ascending_triangle')
                pattern_score += 2
        
        # Bullish Flag
        if len(closes) >= 10:
            # Look for strong upward move followed by consolidation
            price_change = (closes[-10] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
            recent_volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
            
            if price_change > 0.1 and recent_volatility < 0.05:
                patterns_detected.append('bullish_flag')
                pattern_score += 2
        
        # Volume Breakout Pattern
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:-5])
            recent_volume = np.mean(volumes[-5:])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            
            price_breakout = (closes[-1] - closes[-5]) / closes[-5]
            
            if volume_ratio > 1.5 and price_breakout > 0.05:
                patterns_detected.append('volume_breakout')
                pattern_score += 2
        
        # Support/Resistance Bounce
        if len(closes) >= 30:
            # Find potential support/resistance levels
            price_levels = np.concatenate([highs[-30:], lows[-30:]])
            unique_levels = []
            
            for level in price_levels:
                is_unique = True
                for existing in unique_levels:
                    if abs(level - existing) / existing < 0.02:
                        is_unique = False
                        break
                if is_unique:
                    unique_levels.append(level)
            
            current_price = closes[-1]
            for level in unique_levels:
                if abs(current_price - level) / level < 0.01:
                    patterns_detected.append('support_resistance_test')
                    pattern_score += 1
                    break
        
        return {
            'patterns': patterns_detected,
            'pattern_score': pattern_score,
            'pattern_count': len(patterns_detected)
        }
        
    except Exception as e:
        return {'patterns': [], 'pattern_score': 0, 'error': str(e)}

def calculate_advanced_momentum_score(hist_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate advanced momentum score using multiple momentum indicators"""
    try:
        if hist_data is None or len(hist_data) < 50:
            return {'momentum_score': 0, 'momentum_strength': 'weak'}
        
        closes = hist_data['close'].values
        highs = hist_data['high'].values
        lows = hist_data['low'].values
        volumes = hist_data['volume'].values
        
        momentum_score = 0
        
        # Rate of Change (ROC) analysis
        roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) > 5 else 0
        roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) > 10 else 0
        roc_20 = (closes[-1] - closes[-21]) / closes[-21] * 100 if len(closes) > 20 else 0
        
        # Positive ROC scores
        if roc_5 > 2:
            momentum_score += 2
        elif roc_5 > 0:
            momentum_score += 1
        
        if roc_10 > 5:
            momentum_score += 2
        elif roc_10 > 0:
            momentum_score += 1
        
        if roc_20 > 10:
            momentum_score += 3
        elif roc_20 > 0:
            momentum_score += 1
        
        # Williams %R momentum
        if len(closes) >= 14:
            highest_high = np.max(highs[-14:])
            lowest_low = np.min(lows[-14:])
            williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
            
            if williams_r > -20:  # Overbought but strong momentum
                momentum_score += 1
            elif williams_r < -80:  # Oversold, potential reversal
                momentum_score += 2
        
        # Commodity Channel Index (CCI) momentum
        if len(closes) >= 20:
            typical_prices = (highs + lows + closes) / 3
            sma_tp = np.mean(typical_prices[-20:])
            mean_deviation = np.mean(np.abs(typical_prices[-20:] - sma_tp))
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation) if mean_deviation > 0 else 0
            
            if cci > 100:
                momentum_score += 2
            elif cci > 0:
                momentum_score += 1
        
        # Volume-Price Trend (VPT)
        if len(closes) >= 10:
            vpt = 0
            for i in range(1, min(10, len(closes))):
                price_change_pct = (closes[-i] - closes[-i-1]) / closes[-i-1]
                vpt += volumes[-i] * price_change_pct
            
            if vpt > 0:
                momentum_score += 1
        
        # Momentum strength classification
        if momentum_score >= 8:
            strength = 'very_strong'
        elif momentum_score >= 6:
            strength = 'strong'
        elif momentum_score >= 4:
            strength = 'moderate'
        elif momentum_score >= 2:
            strength = 'weak'
        else:
            strength = 'very_weak'
        
        return {
            'momentum_score': momentum_score,
            'momentum_strength': strength,
            'roc_5': roc_5,
            'roc_10': roc_10,
            'roc_20': roc_20,
            'williams_r': williams_r if 'williams_r' in locals() else 0,
            'cci': cci if 'cci' in locals() else 0
        }
        
    except Exception as e:
        return {'momentum_score': 0, 'momentum_strength': 'weak', 'error': str(e)}

def analyze_technicals(
    pair: Dict[str, Any],
    optimized_params: Any,
    config: Dict,
    ml_model: Any,
    ml_model_features: List[str],
    get_historical_data_func: callable,
    calculate_indicators_func: callable,
    log_message: callable
) -> Tuple[Optional[int], Optional[float], Optional[pd.Series]]:
    """Performs advanced technical and ML analysis with multi-timeframe and pattern detection, returning confidence score, ATR, and latest indicator data."""
    symbol = pair['baseToken']['symbol'].upper()
    binance_symbol = f"{symbol}USDT"

    hist_data = get_historical_data_func(binance_symbol, interval=config.get('TA_INTERVAL', '1h'))
    if hist_data is None:
        return None, None, None

    indicators_df = calculate_indicators_func(hist_data)
    if indicators_df is None or indicators_df.empty:
        return None, None, None
    
    last = indicators_df.iloc[-1]
    confidence_score = 0
    
    # Market Regime Analysis
    regime_analysis = detect_market_regime(hist_data)
    if regime_analysis['regime'] in ['strong_uptrend', 'uptrend']:
        regime_bonus = 4 if regime_analysis['regime'] == 'strong_uptrend' else 2
        confidence_score += regime_bonus
        log_message(f"{symbol}: Market regime {regime_analysis['regime']} (+{regime_bonus})", "DEBUG")
    elif regime_analysis['regime'] == 'sideways' and regime_analysis['volatility_regime'] == 'low':
        confidence_score += 1
        log_message(f"{symbol}: Sideways market with low volatility (+1)", "DEBUG")
    
    # Chart Pattern Analysis
    pattern_analysis = detect_chart_patterns(hist_data)
    if pattern_analysis['pattern_score'] > 0:
        confidence_score += pattern_analysis['pattern_score']
        log_message(f"{symbol}: Patterns detected: {', '.join(pattern_analysis['patterns'])} (+{pattern_analysis['pattern_score']})", "DEBUG")
    
    # Advanced Momentum Analysis
    momentum_analysis = calculate_advanced_momentum_score(hist_data)
    if momentum_analysis['momentum_strength'] in ['very_strong', 'strong']:
        momentum_bonus = 3 if momentum_analysis['momentum_strength'] == 'very_strong' else 2
        confidence_score += momentum_bonus
        log_message(f"{symbol}: {momentum_analysis['momentum_strength']} momentum (+{momentum_bonus})", "DEBUG")
    elif momentum_analysis['momentum_strength'] == 'moderate':
        confidence_score += 1
        log_message(f"{symbol}: Moderate momentum (+1)", "DEBUG")
    
    # Multi-timeframe Signal Analysis
    mtf_analysis = analyze_multi_timeframe_signals(symbol, get_historical_data_func, calculate_indicators_func)
    if mtf_analysis.get('recommendation') == 'BUY':
        mtf_bonus = min(3, int(mtf_analysis.get('normalized_signal', 0)))
        confidence_score += mtf_bonus
        log_message(f"{symbol}: Multi-timeframe BUY signal (+{mtf_bonus})", "DEBUG")
    
    # ML Analysis
    confidence_score += _get_ml_bonus(symbol, last, ml_model, ml_model_features, config, log_message)
    
    # Enhanced Traditional Indicator Checks
    ema_short_col = f"EMA_{config.get('TA_EMA_SHORT', 20)}"
    ema_long_col = f"EMA_{config.get('TA_EMA_LONG', 50)}"
    current_price = last.get('close', 0)
    
    # Multi-EMA trend analysis
    if current_price > last[ema_short_col] > last[ema_long_col]:
        confidence_score += 3  # Strong trend alignment
        log_message(f"{symbol}: Strong EMA trend alignment (+3)", "DEBUG")
    elif last[ema_short_col] > last[ema_long_col]:
        confidence_score += 2  # Basic trend confirmation
        log_message(f"{symbol}: EMA trend bullish (+2)", "DEBUG")
    
    # Enhanced RSI analysis
    rsi_col = f"RSI_{config.get('TA_RSI_PERIOD', 14)}"
    rsi_value = last[rsi_col]
    if rsi_value < 25:  # Extremely oversold
        confidence_score += 4
        log_message(f"{symbol}: RSI extremely oversold {rsi_value:.1f} (+4)", "DEBUG")
    elif rsi_value < 30:
        confidence_score += 3
        log_message(f"{symbol}: RSI oversold {rsi_value:.1f} (+3)", "DEBUG")
    elif rsi_value < 40:
        confidence_score += 2
        log_message(f"{symbol}: RSI favorable {rsi_value:.1f} (+2)", "DEBUG")
    elif rsi_value < optimized_params.rsi_overbought * 1.1:
        confidence_score += 1
        log_message(f"{symbol}: RSI acceptable {rsi_value:.1f} (+1)", "DEBUG")
    elif rsi_value > 80:
        confidence_score -= 2  # Penalty for extreme overbought
        log_message(f"{symbol}: RSI extremely overbought {rsi_value:.1f} (-2)", "DEBUG")
    
    # Enhanced MACD analysis
    macd_col = f"MACD_{config.get('TA_MACD_FAST', 12)}_{config.get('TA_MACD_SLOW', 26)}_{config.get('TA_MACD_SIGNAL', 9)}"
    macds_col = f"MACDs_{config.get('TA_MACD_FAST', 12)}_{config.get('TA_MACD_SLOW', 26)}_{config.get('TA_MACD_SIGNAL', 9)}"
    macd_value = last[macd_col]
    macd_signal = last[macds_col]
    
    if macd_value > macd_signal and macd_value > 0:
        confidence_score += 3  # Strong bullish MACD
        log_message(f"{symbol}: MACD strong bullish (+3)", "DEBUG")
    elif macd_value > macd_signal:
        confidence_score += 2  # Bullish crossover
        log_message(f"{symbol}: MACD bullish crossover (+2)", "DEBUG")
    elif macd_value > 0:
        confidence_score += 1  # Above zero line
        log_message(f"{symbol}: MACD above zero (+1)", "DEBUG")
    
    # Enhanced Volume analysis
    volume_ratio = last.get('volume', 0) / last.get('volume_sma_20', 1)
    if volume_ratio > 2.0:
        confidence_score += 3
        log_message(f"{symbol}: Exceptional volume {volume_ratio:.1f}x (+3)", "DEBUG")
    elif volume_ratio > 1.5:
        confidence_score += 2
        log_message(f"{symbol}: High volume {volume_ratio:.1f}x (+2)", "DEBUG")
    elif volume_ratio > 1.2:
        confidence_score += 1
        log_message(f"{symbol}: Above avg volume {volume_ratio:.1f}x (+1)", "DEBUG")
    
    # Enhanced Bollinger Band analysis
    bbl_col = f"BBL_20_2.0"
    bbu_col = f"BBU_20_2.0"
    if bbl_col in last.index and bbu_col in last.index:
        bb_position = (current_price - last[bbl_col]) / (last[bbu_col] - last[bbl_col])
        if bb_position < 0.2:  # Near lower band
            confidence_score += 2
            log_message(f"{symbol}: Near Bollinger lower band (+2)", "DEBUG")
        elif bb_position > 0.8:  # Near upper band - reduce score
            confidence_score -= 1
            log_message(f"{symbol}: Near Bollinger upper band (-1)", "DEBUG")
    elif bbl_col in last.index and current_price < last[bbl_col] * 1.05:  # Within 5% of lower band
        confidence_score += 1
        log_message(f"{symbol}: Near Bollinger support (+1)", "DEBUG")
    
    # Risk assessment and penalties
    risk_penalty = 0
    if regime_analysis['volatility_regime'] == 'high':
        risk_penalty += 1
        log_message(f"{symbol}: High volatility regime (-1)", "DEBUG")
    if rsi_value > 75:
        risk_penalty += 1
        log_message(f"{symbol}: Overbought conditions (-1)", "DEBUG")
    if volume_ratio < 0.8:
        risk_penalty += 1
        log_message(f"{symbol}: Low volume warning (-1)", "DEBUG")
    
    confidence_score -= risk_penalty

    atr_col = f"ATRr_{config.get('ATR_PERIOD', 14)}"
    atr_value = last.get(atr_col, 0)
    
    log_message(f"{symbol}: Final confidence score: {confidence_score} (regime: {regime_analysis['regime']}, patterns: {len(pattern_analysis['patterns'])}, momentum: {momentum_analysis['momentum_strength']})", "INFO")

    return confidence_score, atr_value, last

def calculate_position_size(
    symbol: str,
    atr_value: float,
    confidence_score: int,
    optimized_params: Any,
    config: Dict,
    get_current_portfolio_value_func: callable,
    log_message: callable,
    last_close_price: float
) -> float:
    """Calculates the final trade amount based on risk and sizing strategies."""
    base_trade_amount = config.get('TRADE_USDT', 10)
    trade_amount = base_trade_amount

    if config.get('VOLATILITY_SIZING_ENABLED', False) and atr_value > 0:
        try:
            portfolio_value = get_current_portfolio_value_func()
            risk_per_trade_pct = config.get('RISK_PER_TRADE_PCT', 1.0) / 100.0
            base_atr_multiplier = config.get('ATR_STOP_LOSS_MULTIPLIER', 2.0)
            atr_multiplier = base_atr_multiplier * optimized_params.stop_loss_multiplier
            
            max_trade_size_usd = config.get('MAX_TRADE_SIZE_USDT', portfolio_value * 0.1)
            min_trade_size_usd = config.get('MIN_TRADE_SIZE_USDT', 5)

            dollar_risk = portfolio_value * risk_per_trade_pct
            entry_price = last_close_price
            stop_loss_distance_per_token = atr_value * atr_multiplier
            
            if stop_loss_distance_per_token > 0 and entry_price > 0:
                risk_per_token_pct = stop_loss_distance_per_token / entry_price
                calculated_size_usd = dollar_risk / risk_per_token_pct
                trade_amount = max(min_trade_size_usd, min(calculated_size_usd, max_trade_size_usd))
                log_message(f"Volatility Sizing for {symbol}: Portfolio=${portfolio_value:.0f}, Risk=${dollar_risk:.2f}, ATR=${atr_value:.4f}, Size=${trade_amount:.2f}", "INFO")
        except Exception as e:
            log_message(f"Error in volatility sizing for {symbol}: {e}. Using base trade amount.", "ERROR")
            trade_amount = base_trade_amount
    
    elif config.get('DYNAMIC_SIZING_ENABLED', False):
        # More aggressive position sizing based on confidence score
        # Higher multiplier and lower threshold for bonus calculation
        min_confidence_for_bonus = 1  # Reduced from 2 to be more aggressive
        confidence_bonus_multiplier = config.get('DYNAMIC_SIZING_CONFIDENCE_BONUS', 2.5) * 1.5  # 50% increase
        
        if confidence_score > min_confidence_for_bonus:
            bonus = (confidence_score - min_confidence_for_bonus) * confidence_bonus_multiplier
            # Cap the bonus at 3x the base amount for risk management
            max_bonus = base_trade_amount * 2
            bonus = min(bonus, max_bonus)
            trade_amount += bonus
            log_message(f"Enhanced position sizing for {symbol}: Base=${base_trade_amount:.2f}, Confidence={confidence_score}, Bonus=${bonus:.2f}, Total=${trade_amount:.2f}", "INFO")

    return trade_amount

class TradeOpportunityAnalyzer:
    """Main class for trade opportunity analysis with advanced algorithms"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze_fundamentals(self, pair: Dict[str, Any], open_symbols: set, optimized_params: Any, log_message: callable = None) -> bool:
        """Wrapper for analyze_fundamentals function"""
        return analyze_fundamentals(pair, open_symbols, optimized_params, self.config, log_message)
    
    def analyze_technicals(self, pair: Dict[str, Any], optimized_params: Any, ml_model: Any, ml_model_features: List[str], 
                          get_historical_data_func: callable, calculate_indicators_func: callable, log_message: callable) -> Tuple[Optional[int], Optional[float], Optional[pd.Series]]:
        """Wrapper for analyze_technicals function"""
        return analyze_technicals(pair, optimized_params, self.config, ml_model, ml_model_features, 
                                get_historical_data_func, calculate_indicators_func, log_message)
    
    def calculate_position_size(self, symbol: str, atr_value: float, confidence_score: int, optimized_params: Any,
                               get_current_portfolio_value_func: callable, log_message: callable, last_close_price: float) -> float:
        """Wrapper for calculate_position_size function"""
        return calculate_position_size(symbol, atr_value, confidence_score, optimized_params, self.config,
                                     get_current_portfolio_value_func, log_message, last_close_price)
    
    def detect_market_regime(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Wrapper for detect_market_regime function"""
        return detect_market_regime(hist_data)
    
    def detect_chart_patterns(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Wrapper for detect_chart_patterns function"""
        return detect_chart_patterns(hist_data)
    
    def calculate_advanced_momentum_score(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Wrapper for calculate_advanced_momentum_score function"""
        return calculate_advanced_momentum_score(hist_data)
    
    def analyze_multi_timeframe_signals(self, symbol: str, get_historical_data_func: callable, calculate_indicators_func: callable) -> Dict[str, Any]:
        """Wrapper for analyze_multi_timeframe_signals function"""
        return analyze_multi_timeframe_signals(symbol, get_historical_data_func, calculate_indicators_func)