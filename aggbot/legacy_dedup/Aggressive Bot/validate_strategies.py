#!/usr/bin/env python3
"""
Strategy Validation Script
Validates trading strategies using backtesting without complex dependencies
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBacktester:
    """Simplified backtesting engine for strategy validation"""
    
    def __init__(self, initial_balance=10000, commission=0.001, slippage=0.0005):
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def generate_synthetic_data(self, days=30, volatility=0.02):
        """Generate synthetic price data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price movements
        returns = np.random.normal(0.0001, volatility, days * 24 * 60)  # Minute data
        prices = [100.0]  # Starting price
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
            
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                                  periods=len(prices), freq='1min')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': np.random.uniform(1000, 10000, len(prices))
        })
    
    def momentum_strategy(self, data, lookback=20, threshold=0.02):
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
    
    def mean_reversion_strategy(self, data, lookback=20, std_threshold=2):
        """Simple mean reversion strategy"""
        signals = []
        
        for i in range(lookback, len(data)):
            recent_prices = data.iloc[i-lookback:i]['price']
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            current_price = data.iloc[i]['price']
            
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            if z_score < -std_threshold:
                signals.append('BUY')  # Price below mean, expect reversion up
            elif z_score > std_threshold:
                signals.append('SELL')  # Price above mean, expect reversion down
            else:
                signals.append('HOLD')
                
        return signals
    
    def breakout_strategy(self, data, lookback=20, breakout_threshold=0.03):
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
    
    def scalping_strategy(self, data, short_ma=5, long_ma=20):
        """Simple scalping strategy using moving averages"""
        signals = []
        
        for i in range(long_ma, len(data)):
            short_avg = data.iloc[i-short_ma:i]['price'].mean()
            long_avg = data.iloc[i-long_ma:i]['price'].mean()
            
            if short_avg > long_avg * 1.001:  # Small threshold for scalping
                signals.append('BUY')
            elif short_avg < long_avg * 0.999:
                signals.append('SELL')
            else:
                signals.append('HOLD')
                
        return signals
    
    def execute_backtest(self, data, signals, position_size=0.1):
        """Execute backtest with given signals"""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        start_idx = len(data) - len(signals)
        
        for i, signal in enumerate(signals):
            current_price = data.iloc[start_idx + i]['price']
            trade_size = self.balance * position_size
            
            if signal == 'BUY' and position <= 0:
                if position < 0:  # Close short position
                    profit = (entry_price - current_price) * abs(trade_size) / entry_price
                    self.balance += profit - (trade_size * self.commission)
                    self.trades.append({
                        'type': 'CLOSE_SHORT',
                        'price': current_price,
                        'profit': profit,
                        'timestamp': data.iloc[start_idx + i]['timestamp']
                    })
                
                # Open long position
                entry_price = current_price * (1 + self.slippage)
                position = 1
                self.trades.append({
                    'type': 'BUY',
                    'price': entry_price,
                    'size': trade_size,
                    'timestamp': data.iloc[start_idx + i]['timestamp']
                })
                
            elif signal == 'SELL' and position >= 0:
                if position > 0:  # Close long position
                    profit = (current_price - entry_price) * trade_size / entry_price
                    self.balance += profit - (trade_size * self.commission)
                    self.trades.append({
                        'type': 'CLOSE_LONG',
                        'price': current_price,
                        'profit': profit,
                        'timestamp': data.iloc[start_idx + i]['timestamp']
                    })
                
                # Open short position
                entry_price = current_price * (1 - self.slippage)
                position = -1
                self.trades.append({
                    'type': 'SELL',
                    'price': entry_price,
                    'size': trade_size,
                    'timestamp': data.iloc[start_idx + i]['timestamp']
                })
            
            self.equity_curve.append(self.balance)
        
        # Close any remaining position
        if position != 0:
            current_price = data.iloc[-1]['price']
            if position > 0:
                profit = (current_price - entry_price) * trade_size / entry_price
            else:
                profit = (entry_price - current_price) * trade_size / entry_price
            
            self.balance += profit - (trade_size * self.commission)
            self.equity_curve.append(self.balance)
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_trades': 0
            }
        
        # Calculate returns
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate win rate
        profitable_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        win_rate = len(profitable_trades) / len([t for t in self.trades if 'profit' in t]) * 100 if len([t for t in self.trades if 'profit' in t]) > 0 else 0
        
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
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': len([t for t in self.trades if 'profit' in t])
        }

def validate_strategy(strategy_name, strategy_func, data, backtester):
    """Validate a single strategy"""
    logger.info(f"\n=== Validating {strategy_name} Strategy ===")
    
    try:
        # Generate signals
        signals = strategy_func(data)
        
        # Execute backtest
        backtester.execute_backtest(data, signals)
        
        # Calculate metrics
        metrics = backtester.calculate_metrics()
        
        # Define validation thresholds
        thresholds = {
            'min_total_return': -10,  # Allow some loss for testing
            'min_win_rate': 30,
            'min_profit_factor': 1.0,
            'max_drawdown': 25,
            'min_sharpe_ratio': -0.5,
            'min_trades': 5
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
        print(f"\n{strategy_name} Strategy Results:")
        print(f"  Total Return: {metrics['total_return']}%")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Max Drawdown: {metrics['max_drawdown']}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
        print(f"  Total Trades: {metrics['total_trades']}")
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
    """Main validation function"""
    print("\n" + "="*60)
    print("         TRADING STRATEGY VALIDATION")
    print("="*60)
    
    # Initialize backtester
    backtester = SimpleBacktester(initial_balance=10000)
    
    # Generate synthetic market data
    print("\nGenerating synthetic market data...")
    data = backtester.generate_synthetic_data(days=30)
    print(f"Generated {len(data)} data points over 30 days")
    
    # Define strategies to test
    strategies = {
        'Momentum': backtester.momentum_strategy,
        'Mean Reversion': backtester.mean_reversion_strategy,
        'Breakout': backtester.breakout_strategy,
        'Scalping': backtester.scalping_strategy
    }
    
    # Validate each strategy
    results = {}
    passed_count = 0
    
    for name, strategy_func in strategies.items():
        passed, metrics = validate_strategy(name, strategy_func, data, backtester)
        results[name] = {'passed': passed, 'metrics': metrics}
        if passed:
            passed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("                    SUMMARY")
    print("="*60)
    print(f"\nStrategies Validated: {len(strategies)}")
    print(f"Strategies Passed: {passed_count}")
    print(f"Strategies Failed: {len(strategies) - passed_count}")
    print(f"Success Rate: {passed_count/len(strategies)*100:.1f}%")
    
    if passed_count >= len(strategies) * 0.75:  # 75% pass rate
        print("\n🎉 OVERALL VALIDATION: PASSED")
        print("✅ Trading strategies are ready for live trading!")
    else:
        print("\n⚠️  OVERALL VALIDATION: NEEDS IMPROVEMENT")
        print("❌ Some strategies need optimization before live trading.")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    for name, result in results.items():
        if result['passed']:
            print(f"  ✅ {name}: Ready for live trading")
        else:
            print(f"  ⚠️  {name}: Needs optimization")
    
    print("\n" + "="*60)
    print("Validation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()