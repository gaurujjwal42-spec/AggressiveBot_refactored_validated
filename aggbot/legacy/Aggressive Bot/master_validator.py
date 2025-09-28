#!/usr/bin/env python3
"""
Master Strategy Validation Script
Consolidates all backtesting profiles (Simple, Optimized, Production, etc.)
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

# Import strategies from the centralized strategies file
from strategies import (
    simple_momentum_strategy, simple_mean_reversion_strategy, simple_breakout_strategy,
    simple_scalping_strategy,
    enhanced_momentum_strategy, enhanced_mean_reversion_strategy, enhanced_breakout_strategy,
    enhanced_scalping_strategy,
    production_momentum_strategy, production_mean_reversion_strategy,
    production_breakout_strategy, production_scalping_strategy,
    ultra_momentum_strategy, ultra_mean_reversion_strategy, ultra_breakout_strategy,
    ultra_scalping_strategy,
    AdvancedMomentumStrategy, AdvancedMeanReversionStrategy,
    AdvancedBreakoutStrategy, AdvancedScalpingStrategy
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionBacktester:
    """
    Backtesting engine for production-level validation.
    Features realistic market simulation and comprehensive technical indicators.
    """
    def __init__(self, initial_balance=10000, commission=0.0002, slippage=0.0001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.max_position_size = 0.04  # Conservative 4% per trade
        self.stop_loss = 0.015         # 1.5% stop loss
        self.take_profit = 0.03        # 3% take profit
        
    def generate_production_data(self, days=90, base_price=100.0):
        """Generate production-quality synthetic data with realistic market behavior"""
        np.random.seed(42)  # For reproducible results
        
        # Create realistic market cycles
        total_minutes = days * 24 * 60
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                                  periods=total_minutes, freq='1min')
        
        # Generate realistic price movements with multiple cycles
        prices = [base_price]
        trend_strength = 0.0003
        volatility = 0.012
        
        # Create market phases: trending up, sideways, trending down, recovery
        phase_length = total_minutes // 8
        phases = ['up_trend', 'sideways', 'down_trend', 'recovery', 
                 'up_trend', 'sideways', 'down_trend', 'recovery']
        
        for i in range(total_minutes):
            phase_idx = min(i // phase_length, len(phases) - 1)
            current_phase = phases[phase_idx]
            
            # Adjust trend based on phase
            if current_phase == 'up_trend':
                trend = trend_strength * 1.5
                vol = volatility * 0.8
            elif current_phase == 'down_trend':
                trend = -trend_strength * 1.2
                vol = volatility * 1.1
            elif current_phase == 'recovery':
                trend = trend_strength * 2.0
                vol = volatility * 0.9
            else:  # sideways
                trend = trend_strength * 0.1
                vol = volatility * 0.7
            
            # Add some noise and mean reversion
            noise = np.random.normal(0, vol)
            mean_reversion = (base_price - prices[-1]) * 0.0001
            
            price_change = trend + noise + mean_reversion
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Create DataFrame with technical indicators
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices[1:],  # Remove initial price
            'volume': np.random.uniform(5000, 20000, len(timestamps))
        })
        
        # Add comprehensive technical indicators
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['price'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic
        low_14 = df['price'].rolling(window=14).min()
        high_14 = df['price'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['price'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ATR (Average True Range)
        df['atr'] = df['price'].rolling(window=14).std() * np.sqrt(14)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    # This method is kept for the 'production' profile
    def execute_production_backtest(self, data, signals, position_size=0.04):
        """Execute production-ready backtest"""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        position = 0
        entry_price = 0
        consecutive_losses = 0
        max_consecutive_losses = 3
        
        start_idx = len(data) - len(signals)
        if start_idx < 0:
            logger.error("Signal array is longer than data array. Aborting backtest.")
            return

        for i, signal in enumerate(signals):
            current_price = data.iloc[start_idx + i]['price']
            current_time = data.iloc[start_idx + i]['timestamp']
            
            # Dynamic position sizing
            if consecutive_losses >= 2:
                current_position_size = position_size * 0.5
            else:
                current_position_size = position_size
            
            trade_size = self.balance * current_position_size
            
            # Risk management
            if position != 0 and entry_price > 0:
                if position > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        profit = pnl_pct * trade_size
                        self.balance += profit - (trade_size * self.commission)
                        
                        self.trades.append({
                            'type': 'CLOSE_LONG',
                            'price': current_price,
                            'profit': profit,
                            'timestamp': current_time
                        })
                        
                        if profit < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        
                        position = 0
                        entry_price = 0
                        continue
                
                elif position < 0:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        profit = pnl_pct * trade_size
                        self.balance += profit - (trade_size * self.commission)
                        
                        self.trades.append({
                            'type': 'CLOSE_SHORT',
                            'price': current_price,
                            'profit': profit,
                            'timestamp': current_time
                        })
                        
                        if profit < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        
                        position = 0
                        entry_price = 0
                        continue
            
            # Skip trading if too many losses
            if consecutive_losses >= max_consecutive_losses:
                continue
            
            # Execute signals
            if signal == 'BUY' and position <= 0:
                if position < 0:  # Close short
                    profit = (entry_price - current_price) * trade_size / entry_price
                    self.balance += profit - (trade_size * self.commission)
                    self.trades.append({
                        'type': 'CLOSE_SHORT',
                        'price': current_price,
                        'profit': profit,
                        'timestamp': current_time
                    })
                    
                    if profit < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                
                # Open long
                entry_price = current_price * (1 + self.slippage)
                position = 1
                self.trades.append({
                    'type': 'BUY',
                    'price': entry_price,
                    'size': trade_size,
                    'timestamp': current_time
                })
            
            elif signal == 'SELL' and position >= 0:
                if position > 0:  # Close long
                    profit = (current_price - entry_price) * trade_size / entry_price
                    self.balance += profit - (trade_size * self.commission)
                    self.trades.append({
                        'type': 'CLOSE_LONG',
                        'price': current_price,
                        'profit': profit,
                        'timestamp': current_time
                    })
                    
                    if profit < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                
                # Open short
                entry_price = current_price * (1 - self.slippage)
                position = -1
                self.trades.append({
                    'type': 'SELL',
                    'price': entry_price,
                    'size': trade_size,
                    'timestamp': current_time
                })
            
            self.equity_curve.append(self.balance)
        
        # Close remaining position
        if position != 0:
            current_price = data.iloc[-1]['price']
            if position > 0:
                profit = (current_price - entry_price) * trade_size / entry_price
            else:
                profit = (entry_price - current_price) * trade_size / entry_price
            
            self.balance += profit - (trade_size * self.commission)
            self.equity_curve.append(self.balance)
    
    # This method is kept for the 'production' profile
    def calculate_production_metrics(self):
        """Calculate production-ready performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
                'avg_profit_per_trade': 0
            }
        
        # Calculate returns
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate win rate
        profitable_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        total_closed_trades = [t for t in self.trades if 'profit' in t]
        win_rate = len(profitable_trades) / len(total_closed_trades) * 100 if len(total_closed_trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([t['profit'] for t in self.trades if t.get('profit', 0) > 0])
        gross_loss = abs(sum([t['profit'] for t in self.trades if t.get('profit', 0) < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Calculate max drawdown
        peak = self.initial_balance
        max_drawdown = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average profit per trade
        avg_profit_per_trade = sum([t.get('profit', 0) for t in self.trades]) / len(total_closed_trades) if len(total_closed_trades) > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': len(total_closed_trades),
            'avg_profit_per_trade': round(avg_profit_per_trade, 2)
        }

# --- Add other backtesters from other files ---

class SimpleBacktester(ProductionBacktester):
    """Simplified backtesting engine for basic strategy validation"""
    def __init__(self, initial_balance=10000, commission=0.001, slippage=0.0005):
        super().__init__(initial_balance, commission, slippage)

    def generate_simple_data(self, days=30, volatility=0.02):
        np.random.seed(42)
        returns = np.random.normal(0.0001, volatility, days * 24 * 60)
        prices = [100.0]
        for ret in returns:
            prices.append(max(prices[-1] * (1 + ret), 0.01))
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), periods=len(prices), freq='1min')
        return pd.DataFrame({'timestamp': timestamps, 'price': prices, 'volume': np.random.uniform(1000, 10000, len(prices))})

class OptimizedBacktester(ProductionBacktester):
    """Optimized backtesting engine with enhanced risk management"""
    def __init__(self, initial_balance=10000, commission=0.0005, slippage=0.0002):
        super().__init__(initial_balance, commission, slippage)
        self.max_position_size = 0.05
        self.stop_loss = 0.02
        self.take_profit = 0.04

    def generate_optimized_data(self, days=30, volatility=0.015):
        np.random.seed(42)
        base_trend = 0.0002
        returns = np.random.normal(base_trend, volatility, days * 24 * 60)
        prices = [100.0]
        for ret in returns:
            prices.append(max(prices[-1] * (1 + ret), 0.01))
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), periods=len(prices), freq='1min')
        df = pd.DataFrame({'timestamp': timestamps, 'price': prices, 'volume': np.random.uniform(1000, 10000, len(prices))})
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        return df

class UltraOptimizedBacktester(ProductionBacktester):
    """Ultra-optimized backtesting engine with aggressive parameters"""
    def __init__(self, initial_balance=10000, commission=0.0003, slippage=0.0001):
        super().__init__(initial_balance, commission, slippage)
        self.max_position_size = 0.10
        self.stop_loss = 0.02
        self.take_profit = 0.07

    def generate_ultra_data(self, days=60, volatility=0.018):
        np.random.seed(42)
        base_trend = 0.0005
        trend_changes = np.random.choice([-1, 1], size=days//10) * 0.0003
        returns = []
        current_trend = base_trend
        for day in range(days):
            if day % 10 == 0 and day > 0:
                current_trend += trend_changes[min(day // 10 - 1, len(trend_changes) - 1)]
            daily_returns = np.random.normal(current_trend, volatility, 24 * 60)
            returns.extend(daily_returns)
        prices = [100.0]
        for ret in returns:
            prices.append(max(prices[-1] * (1 + ret), 0.01))
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), periods=len(prices), freq='1min')
        df = pd.DataFrame({'timestamp': timestamps, 'price': prices, 'volume': np.random.uniform(2000, 15000, len(prices))})
        for p in [3, 5, 10, 20, 50]: df[f'sma_{p}'] = df['price'].rolling(window=p).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        for p in [7, 14, 21]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss
            df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        for p in [15, 20, 25]:
            df[f'bb_middle_{p}'] = df['price'].rolling(window=p).mean()
            bb_std = df['price'].rolling(window=p).std()
            df[f'bb_upper_{p}'] = df[f'bb_middle_{p}'] + (bb_std * 2)
            df[f'bb_lower_{p}'] = df[f'bb_middle_{p}'] - (bb_std * 2)
        df['volatility'] = df['price'].rolling(window=20).std()
        return df

class FinalBacktester(ProductionBacktester):
    """Advanced backtesting with realistic market conditions"""
    def __init__(self, initial_balance=10000):
        super().__init__(initial_balance)

    def generate_final_data(self, days=365, volatility=0.02):
        np.random.seed(42)
        returns = []
        current_regime = 'normal'
        regime_duration = 0
        for i in range(days * 24):
            if regime_duration <= 0:
                current_regime = np.random.choice(['normal', 'trending', 'volatile'], p=[0.6, 0.25, 0.15])
                regime_duration = np.random.randint(24, 168)
            regime_duration -= 1
            if current_regime == 'normal':
                ret = np.random.normal(0, volatility)
            elif current_regime == 'trending':
                trend = 0.0005 if np.random.random() > 0.5 else -0.0005
                ret = np.random.normal(trend, volatility * 0.8)
            else:
                ret = np.random.normal(0, volatility * 2.5)
            if i > 0 and abs(returns[-1]) > volatility:
                ret *= 1.5
            returns.append(ret)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        data = []
        for i in range(0, len(prices)-4, 4):
            if i + 4 < len(prices):
                candle_prices = prices[i:i+4]
                data.append({
                    'timestamp': datetime.now() - timedelta(hours=len(data)*4),
                    'open': candle_prices[0], 'high': max(candle_prices),
                    'low': min(candle_prices), 'close': candle_prices[-1],
                    'volume': np.random.uniform(1000, 10000)
                })
        df = pd.DataFrame(data).set_index('timestamp').sort_index()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        return df

    def execute_final_backtest(self, df, strategy):
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        signals = strategy.generate_signals(df)
        position = 0
        entry_price = 0
        for i, (signal, row) in enumerate(zip(signals, df.itertuples())):
            current_price = row.close
            portfolio_value = self.balance # Simplified for this context
            risk_amount = portfolio_value * 0.02
            atr = getattr(row, 'atr', current_price * 0.02) if hasattr(row, 'atr') else current_price * 0.02
            position_size = min(risk_amount / (atr * 2), portfolio_value * 0.1 / current_price)
            trade_size = position_size * current_price

            if signal == 'buy' and position <= 0:
                if position < 0: # Close short
                    profit = (entry_price - current_price) * trade_size / entry_price
                    self.balance += profit
                position = 1
                entry_price = current_price
            elif signal == 'sell' and position >= 0:
                if position > 0: # Close long
                    profit = (current_price - entry_price) * trade_size / entry_price
                    self.balance += profit
                position = -1
                entry_price = current_price
            self.equity_curve.append(self.balance)


def validate_strategy(strategy_name, strategy_func, data, backtester, profile):
    """Validate a production-ready strategy"""
    logger.info(f"\n=== Validating {strategy_name} Strategy (Profile: {profile}) ===")
    
    try:
        # Generate signals
        if isinstance(strategy_func, type): # It's a class for 'final' profile
            signals = strategy_func().generate_signals(data)
        else:
            signals = strategy_func(data)
        
        # Execute backtest
        # Use a generic backtest execution method
        backtester.execute_production_backtest(data, signals) # Assuming this is the most generic one
        
        # Calculate metrics
        metrics = backtester.calculate_production_metrics()
        
        # Define realistic production thresholds
        thresholds = {
            'min_total_return': 1.0,   # 1% minimum return
            'min_win_rate': 35,        # 35% minimum win rate
            'min_profit_factor': 1.02, # 1.02 minimum profit factor
            'max_drawdown': 25,        # 25% maximum drawdown
            'min_sharpe_ratio': 0.1,   # 0.1 minimum Sharpe ratio
            'min_trades': 5            # 5 minimum trades
        }
        
        # Check validation
        passed = (
            metrics['total_return'] >= thresholds['min_total_return'] and
            metrics['win_rate'] >= thresholds['min_win_rate'] and
            metrics['profit_factor'] >= thresholds['min_profit_factor'] and
            metrics['max_drawdown'] <= thresholds['max_drawdown'] and
            metrics['sharpe_ratio'] >= thresholds['min_sharpe_ratio'] and
            metrics['total_trades'] >= thresholds['min_trades']
        )
        
        # Print results
        print(f"\n{strategy_name} Strategy Results (Profile: {profile}):")
        print(f"  Total Return: {metrics['total_return']}%")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Max Drawdown: {metrics['max_drawdown']}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Avg Profit/Trade: ${metrics['avg_profit_per_trade']:.2f}")
        print(f"  Validation: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        if not passed:
            print("  Issues:")
            if metrics['total_return'] < thresholds['min_total_return']:
                print(f"    - Total return too low: {metrics['total_return']}% < {thresholds['min_total_return']}%")
            if metrics['win_rate'] < thresholds['min_win_rate']:
                print(f"    - Win rate too low: {metrics['win_rate']}% < {thresholds['min_win_rate']}%")
            if metrics['profit_factor'] < thresholds['min_profit_factor']:
                print(f"    - Profit factor too low: {metrics['profit_factor']} < {thresholds['min_profit_factor']}")
            if metrics['max_drawdown'] > thresholds['max_drawdown']:
                print(f"    - Max drawdown too high: {metrics['max_drawdown']}% > {thresholds['max_drawdown']}%")
            if metrics['sharpe_ratio'] < thresholds['min_sharpe_ratio']:
                print(f"    - Sharpe ratio too low: {metrics['sharpe_ratio']} < {thresholds['min_sharpe_ratio']}")
            if metrics['total_trades'] < thresholds['min_trades']:
                print(f"    - Not enough trades: {metrics['total_trades']} < {thresholds['min_trades']}")
        
        return passed, metrics
        
    except Exception as e:
        logger.error(f"Error validating {strategy_name}: {e}")
        print(f"\n{strategy_name} Strategy: ❌ ERROR - {e}")
        return False, {}

def main():
    """Main validation function for all strategy profiles"""
    parser = argparse.ArgumentParser(description="Master Strategy Validation Script.")
    parser.add_argument(
        '--profile',
        type=str,
        choices=['simple', 'optimized', 'ultra', 'production', 'final', 'all'],
        default='all',
        help='Specify which validation profile to run (default: all).'
    )
    parser.add_argument(
        '--strategy', 
        type=str, 
        choices=['momentum', 'mean_reversion', 'breakout', 'scalping', 'all'], 
        default='all',
        help='Specify which strategy to backtest within the profile (default: all).'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=90,
        help='Number of days for synthetic data generation (default: 120).'
    )
    args = parser.parse_args()

    print("\n" + "="*85)
    print("                MASTER STRATEGY VALIDATION")
    print("="*85)

    profiles = {
        'simple': {
            'backtester': SimpleBacktester(),
            'data_gen': 'generate_simple_data',
            'strategies': {
                'Momentum': simple_momentum_strategy,
                'Mean Reversion': simple_mean_reversion_strategy,
                'Breakout': simple_breakout_strategy,
                'Scalping': simple_scalping_strategy,
            }
        },
        'optimized': {
            'backtester': OptimizedBacktester(),
            'data_gen': 'generate_optimized_data',
            'strategies': {
                'Enhanced Momentum': enhanced_momentum_strategy,
                'Enhanced Mean Reversion': enhanced_mean_reversion_strategy,
                'Enhanced Breakout': enhanced_breakout_strategy,
                'Enhanced Scalping': enhanced_scalping_strategy,
            }
        },
        'ultra': {
            'backtester': UltraOptimizedBacktester(),
            'data_gen': 'generate_ultra_data',
            'strategies': {
                'Ultra Momentum': ultra_momentum_strategy,
                'Ultra Mean Reversion': ultra_mean_reversion_strategy,
                'Ultra Breakout': ultra_breakout_strategy,
                'Ultra Scalping': ultra_scalping_strategy,
            }
        },
        'production': {
            'backtester': ProductionBacktester(),
            'data_gen': 'generate_production_data',
            'strategies': {
                'Production Momentum': production_momentum_strategy,
                'Production Mean Reversion': production_mean_reversion_strategy,
                'Production Breakout': production_breakout_strategy,
                'Production Scalping': production_scalping_strategy,
            }
        },
        'final': {
            'backtester': FinalBacktester(),
            'data_gen': 'generate_final_data',
            'strategies': {
                'Advanced Momentum': AdvancedMomentumStrategy,
                'Advanced Mean Reversion': AdvancedMeanReversionStrategy,
                'Advanced Breakout': AdvancedBreakoutStrategy,
                'Advanced Scalping': AdvancedScalpingStrategy,
            }
        }
    }

    profiles_to_run = profiles.keys() if args.profile == 'all' else [args.profile]

    results = {}
    total_passed = 0
    total_validated = 0

    for profile_name in profiles_to_run:
        profile = profiles[profile_name]
        backtester = profile['backtester']
        
        print(f"\n--- Running Profile: {profile_name.upper()} ---")
        data = getattr(backtester, profile['data_gen'])(days=args.days)
        print(f"Generated {len(data)} data points over {args.days} days for profile '{profile_name}'")

        strategies_to_test = profile['strategies']
        if args.strategy != 'all':
            strategies_to_test = {k: v for k, v in strategies_to_test.items() if args.strategy.lower() in k.lower()}

        for name, strategy_func in strategies_to_test.items():
            passed, metrics = validate_strategy(name, strategy_func, data, backtester, profile_name)
            results[f"{profile_name}_{name}"] = {'passed': passed, 'metrics': metrics}
            if passed:
                total_passed += 1
            total_validated += 1

    # Summary
    print("\n" + "="*85)
    print("                     OVERALL VALIDATION SUMMARY")
    print("="*85)
    print(f"\nTotal Strategies Validated: {total_validated}")
    print(f"Total Strategies Passed: {total_passed}")
    print(f"Overall Success Rate: {total_passed/total_validated*100:.1f}%" if total_validated else "N/A")
    
    if total_passed > 0:
        print("\n🎉 VALIDATION COMPLETED WITH SUCCESSFUL STRATEGIES!")
        
        # Save successful strategies configuration
        successful_strategies = {name: result for name, result in results.items() if result['passed']}
        config = {
            'timestamp': datetime.now().isoformat(),
            'validation_profile': args.profile,
            'successful_strategies': list(successful_strategies.keys()),
            'validation_results': successful_strategies,
        }
        
        with open('master_validation_results.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print("💾 Successful strategies saved to master_validation_results.json")
        
    else:
        print("\n⚠️  VALIDATION COMPLETED, BUT NO STRATEGIES PASSED.")
        print("❌ All tested strategies failed validation.")
    
    # Detailed recommendations
    print("\n📋 DETAILED VALIDATION RESULTS:")
    for name, result in results.items():
        profile, strat_name = name.split('_', 1)
        if result['passed']:
            metrics = result['metrics']
            print(f"  ✅ [{profile.upper()}] {strat_name}: PASSED")
            print(f"     - Return: {metrics['total_return']}%, Win Rate: {metrics['win_rate']}%")
            print(f"     - Profit Factor: {metrics['profit_factor']}, Drawdown: {metrics['max_drawdown']}%")
        else:
            print(f"  ❌ [{profile.upper()}] {strat_name}: FAILED")
    
    print("\n" + "="*85)
    print("Master validation completed!")
    print("="*85)
    
    return total_passed > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 MASTER VALIDATION COMPLETE - AT LEAST ONE STRATEGY PASSED!")
    else:
        print("\n🔧 MASTER VALIDATION FAILED - ALL STRATEGIES NEED MORE WORK.")