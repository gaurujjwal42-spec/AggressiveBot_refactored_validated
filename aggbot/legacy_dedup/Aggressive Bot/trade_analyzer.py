#!/usr/bin/env python3
"""
Comprehensive Trade Analysis System
Tracks and analyzes the first 100 executed trades for performance validation
"""

import os
import json
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Individual trade record"""
    trade_id: int
    timestamp: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    trade_amount_usd: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    duration_minutes: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    exit_reason: Optional[str]  # 'tp', 'sl', 'manual', 'timeout'
    fees: float
    slippage: float
    volatility: float
    risk_score: float
    is_winning: Optional[bool]
    drawdown_at_entry: float
    portfolio_value_at_entry: float
    
class TradeAnalyzer:
    """Comprehensive trade analysis and performance tracking"""
    
    def __init__(self, target_trades: int = 100):
        self.target_trades = target_trades
        self.trades: List[TradeRecord] = []
        self.analysis_file = 'trade_analysis_results.json'
        self.trades_file = 'first_100_trades.json'
        self.start_time = datetime.now()
        self.initial_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_pnl_percentage': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'average_trade_duration': 0.0,
            'total_fees': 0.0,
            'average_slippage': 0.0,
            'risk_adjusted_return': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'daily_returns': [],
            'monthly_returns': [],
            'volatility': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        # Load existing data if available
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing trade data and analysis"""
        try:
            with open(self.trades_file, 'r') as f:
                trades_data = json.load(f)
                self.trades = [TradeRecord(**trade) for trade in trades_data]
                self.metrics['total_trades'] = len(self.trades)
                print(f"Loaded {len(self.trades)} existing trades")
        except FileNotFoundError:
            print("No existing trade data found. Starting fresh.")
        except Exception as e:
            print(f"Error loading trade data: {e}")
    
    def add_trade(self, trade_data: Dict[str, Any]) -> Optional[TradeRecord]:
        """Add a new trade record with robust validation."""
        if len(self.trades) >= self.target_trades:
            logger.info(f"Target trade count ({self.target_trades}) reached. No new trades will be added to the analysis.")
            return None

        try:
            # 1. Validate that all required fields are present in the input dictionary.
            required_fields = ['symbol', 'side', 'entry_price', 'quantity', 'trade_amount_usd']
            missing_fields = [field for field in required_fields if field not in trade_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # 2. Create the TradeRecord, performing type casting and validation.
            # This centralizes data conversion and helps catch type/value errors early.
            trade = TradeRecord(
                trade_id=len(self.trades) + 1,
                timestamp=str(trade_data.get('timestamp', datetime.now().isoformat())),
                symbol=str(trade_data['symbol']),
                side=str(trade_data['side']),
                entry_price=float(trade_data['entry_price']),
                exit_price=float(trade_data['exit_price']) if 'exit_price' in trade_data and trade_data['exit_price'] is not None else None,
                quantity=float(trade_data['quantity']),
                trade_amount_usd=float(trade_data['trade_amount_usd']),
                pnl=float(trade_data['pnl']) if 'pnl' in trade_data and trade_data['pnl'] is not None else None,
                pnl_percentage=float(trade_data['pnl_percentage']) if 'pnl_percentage' in trade_data and trade_data['pnl_percentage'] is not None else None,
                duration_minutes=float(trade_data['duration_minutes']) if 'duration_minutes' in trade_data and trade_data['duration_minutes'] is not None else None,
                stop_loss=float(trade_data['stop_loss']) if 'stop_loss' in trade_data and trade_data['stop_loss'] is not None else None,
                take_profit=float(trade_data['take_profit']) if 'take_profit' in trade_data and trade_data['take_profit'] is not None else None,
                exit_reason=str(trade_data['exit_reason']) if 'exit_reason' in trade_data and trade_data['exit_reason'] is not None else None,
                fees=float(trade_data.get('fees', 0.0)),
                slippage=float(trade_data.get('slippage', 0.0)),
                volatility=float(trade_data.get('volatility', 0.0)),
                risk_score=float(trade_data.get('risk_score', 0.0)),
                is_winning=bool(trade_data['is_winning']) if 'is_winning' in trade_data and trade_data['is_winning'] is not None else None,
                drawdown_at_entry=float(trade_data.get('drawdown_at_entry', 0.0)),
                portfolio_value_at_entry=float(trade_data.get('portfolio_value_at_entry', 0.0))
            )

            # 3. Perform additional validation on the created record's values.
            if trade.side not in ['buy', 'sell']:
                raise ValueError(f"Invalid 'side' value: '{trade.side}'. Must be 'buy' or 'sell'.")

            # 4. If validation passes, add the trade and save.
            self.trades.append(trade)
            self.save_trades()

            if trade.pnl is not None:
                self.update_metrics()
            return trade

        except (ValueError, TypeError) as e:
            print(f"ERROR: Could not add trade due to invalid data: {e}. Data provided: {trade_data}")
            return None

    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """Update an existing trade with exit information"""
        try:
            trade_index = trade_id - 1
            if 0 <= trade_index < len(self.trades):
                trade = self.trades[trade_index]
                
                # Update trade fields
                for key, value in update_data.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                
                self.save_trades()
                self.update_metrics()
                return True
                
        except Exception as e:
            print(f"Error updating trade {trade_id}: {e}")
        
        return False
    
    def update_performance_metrics(self, performance_data: Dict[str, Any]):
        """Update performance metrics with runtime data from main loop"""
        try:
            self.metrics.update({
                'runtime_hours': performance_data.get('runtime_hours', 0),
                'cycle_rate': performance_data.get('cycle_rate', 0),
                'success_rate': performance_data.get('success_rate', 0),
                'avg_cycle_time': performance_data.get('avg_cycle_time', 0),
                'last_performance_update': datetime.now().isoformat()
            })
            
            # Save updated metrics
            self.save_analysis()
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
    
    def update_metrics(self):
        """Calculate comprehensive performance metrics"""
        closed_trades = [t for t in self.trades if t.pnl is not None]
        
        if not closed_trades:
            return
        
        # Basic metrics
        self.metrics['total_trades'] = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.is_winning]
        losing_trades = [t for t in closed_trades if not t.is_winning]
        
        self.metrics['winning_trades'] = len(winning_trades)
        self.metrics['losing_trades'] = len(losing_trades)
        self.metrics['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        # P&L metrics
        self.metrics['total_pnl'] = sum(t.pnl for t in closed_trades)
        self.metrics['total_pnl_percentage'] = sum(t.pnl_percentage for t in closed_trades)
        
        if winning_trades:
            self.metrics['average_win'] = statistics.mean(t.pnl for t in winning_trades)
        if losing_trades:
            self.metrics['average_loss'] = statistics.mean(t.pnl for t in losing_trades)
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        self.metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        
        # Risk metrics
        self._calculate_drawdown()
        self._calculate_sharpe_ratio()
        self._calculate_other_ratios()
        
        # Trade duration
        durations = [t.duration_minutes for t in closed_trades if t.duration_minutes]
        if durations:
            self.metrics['average_trade_duration'] = statistics.mean(durations)
        
        # Fees and slippage
        self.metrics['total_fees'] = sum(t.fees for t in closed_trades)
        self.metrics['average_slippage'] = statistics.mean(t.slippage for t in closed_trades)
        
        # Consecutive wins/losses
        self._calculate_consecutive_trades()
        
        # Save updated metrics
        self.save_analysis()
    
    def _calculate_drawdown(self):
        """Calculate maximum and current drawdown"""
        closed_trades = [t for t in self.trades if t.pnl_percentage is not None]
        if not closed_trades:
            return
        
        # Use portfolio value at entry if available, otherwise simulate from PnL
        if all(t.portfolio_value_at_entry > 0 for t in closed_trades):
            equity_curve = np.array([t.portfolio_value_at_entry for t in closed_trades])
        else:
            # Fallback to simulating with PnL if portfolio value is missing
            initial_value = self.initial_portfolio_value or 10000
            equity_curve = np.array([initial_value] + [trade.pnl for trade in closed_trades]).cumsum()

        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown is (running_max - equity_curve) / running_max
        drawdowns = (running_max - equity_curve) / running_max
        
        self.metrics['max_drawdown'] = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
        self.metrics['current_drawdown'] = drawdowns[-1] * 100 if len(drawdowns) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        returns = [t.pnl_percentage for t in self.trades if t.pnl_percentage is not None]
        
        if len(returns) < 2:
            return
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return > 0:
            self.metrics['sharpe_ratio'] = mean_return / std_return
            self.metrics['volatility'] = std_return
    
    def _calculate_other_ratios(self):
        """Calculate Calmar and Sortino ratios"""
        returns = [t.pnl_percentage for t in self.trades if t.pnl_percentage is not None]
        
        if not returns:
            return
        
        mean_return = statistics.mean(returns)
        
        # Calmar ratio
        if self.metrics['max_drawdown'] > 0:
            self.metrics['calmar_ratio'] = mean_return / self.metrics['max_drawdown']
        
        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) >= 2:
            downside_deviation = statistics.stdev(negative_returns)
            if downside_deviation > 0:
                self.metrics['sortino_ratio'] = mean_return / downside_deviation
    
    def _calculate_consecutive_trades(self):
        """Calculate consecutive wins and losses"""
        closed_trades = [t for t in self.trades if t.is_winning is not None]
        if not closed_trades:
            return

        # Calculate max consecutive streaks
        max_wins = 0
        max_losses = 0
        c_wins = 0
        c_losses = 0
        for trade in closed_trades:
            if trade.is_winning:
                c_wins += 1
                c_losses = 0
            else:
                c_losses += 1
                c_wins = 0
            max_wins = max(max_wins, c_wins)
            max_losses = max(max_losses, c_losses)
        
        self.metrics['max_consecutive_wins'] = max_wins
        self.metrics['max_consecutive_losses'] = max_losses
        
        # The current consecutive streak is a real-time metric best tracked by the MonitoringDashboard.
        # This analyzer focuses on historical stats. We'll set the metrics to the final calculated
        # streak from the historical data.
        self.metrics['consecutive_wins'] = c_wins
        self.metrics['consecutive_losses'] = c_losses
    
    def save_trades(self):
        """Save trades to file atomically."""
        try:
            trades_data = [asdict(trade) for trade in self.trades]
            temp_file = self.trades_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
            os.replace(temp_file, self.trades_file)
        except Exception as e:
            print(f"Error saving trades: {e}")
    
    def save_analysis(self):
        """Save analysis results to file atomically."""
        try:
            analysis_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'target_trades': self.target_trades,
                'trades_completed': len([t for t in self.trades if t.pnl is not None]),
                'trades_open': len([t for t in self.trades if t.pnl is None]),
                'analysis_period_days': (datetime.now() - self.start_time).days,
                'metrics': self.metrics,
                'recommendations': self.generate_recommendations()
            }
            
            temp_file = self.analysis_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            os.replace(temp_file, self.analysis_file)
                
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    def generate_recommendations(self) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Win rate analysis
        if self.metrics['win_rate'] < 50:
            recommendations.append("⚠️ Win rate below 50% - Consider tightening entry criteria")
        elif self.metrics['win_rate'] > 70:
            recommendations.append("✅ Excellent win rate - Current strategy is performing well")
        
        # Profit factor analysis
        if self.metrics['profit_factor'] < 1.0:
            recommendations.append("🔴 Profit factor below 1.0 - Strategy is losing money")
        elif self.metrics['profit_factor'] > 1.5:
            recommendations.append("✅ Strong profit factor - Good risk/reward ratio")
        
        # Drawdown analysis
        if self.metrics['max_drawdown'] > 20:
            recommendations.append("⚠️ High maximum drawdown - Consider reducing position sizes")
        
        # Sharpe ratio analysis
        if self.metrics['sharpe_ratio'] < 1.0:
            recommendations.append("📊 Low Sharpe ratio - Returns not compensating for risk")
        elif self.metrics['sharpe_ratio'] > 2.0:
            recommendations.append("✅ Excellent risk-adjusted returns")
        
        # Consecutive losses
        if self.metrics['consecutive_losses'] > 5:
            recommendations.append("🛑 High consecutive losses - Consider pausing trading")
        
        return recommendations
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current progress towards 100 trades"""
        completed_trades = len([t for t in self.trades if t.pnl is not None])
        open_trades = len([t for t in self.trades if t.pnl is None])
        
        return {
            'target_trades': self.target_trades,
            'completed_trades': completed_trades,
            'open_trades': open_trades,
            'total_trades': len(self.trades),
            'progress_percentage': (completed_trades / self.target_trades) * 100,
            'estimated_completion': self._estimate_completion_time(),
            'current_metrics': self.metrics
        }
    
    def _estimate_completion_time(self) -> str:
        """Estimate when 100 trades will be completed"""
        completed_trades = len([t for t in self.trades if t.pnl is not None])
        
        if completed_trades < 5:
            return "Insufficient data for estimation"
        
        elapsed_days = (datetime.now() - self.start_time).days
        if elapsed_days == 0:
            elapsed_days = 1
        
        trades_per_day = completed_trades / elapsed_days
        remaining_trades = self.target_trades - completed_trades
        
        if trades_per_day > 0:
            days_remaining = remaining_trades / trades_per_day
            completion_date = datetime.now() + timedelta(days=days_remaining)
            return completion_date.strftime("%Y-%m-%d")
        
        return "Unable to estimate"
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        progress = self.get_progress_report()
        
        print("\n" + "="*60)
        print("📊 TRADE ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        print(f"\n📈 PROGRESS:")
        print(f"   Target Trades: {progress['target_trades']}")
        print(f"   Completed: {progress['completed_trades']}")
        print(f"   Open: {progress['open_trades']}")
        print(f"   Progress: {progress['progress_percentage']:.1f}%")
        print(f"   Est. Completion: {progress['estimated_completion']}")
        
        print(f"\n💰 PERFORMANCE METRICS:")
        print(f"   Win Rate: {self.metrics['win_rate']:.1f}%")
        print(f"   Total P&L: ${self.metrics['total_pnl']:.2f}")
        print(f"   Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {self.metrics['max_drawdown']:.1f}%")
        print(f"   Average Win: ${self.metrics['average_win']:.2f}")
        print(f"   Average Loss: ${self.metrics['average_loss']:.2f}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        recommendations = self.generate_recommendations()
        if recommendations:
            for rec in recommendations:
                print(f"   {rec}")
        else:
            print("   ✅ No specific recommendations - strategy performing well")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the current performance metrics."""
        return self.metrics
        
        print("\n" + "="*60)

# Global analyzer instance
analyzer = None

def get_trade_analyzer() -> TradeAnalyzer:
    """Get the global trade analyzer instance"""
    global analyzer
    if analyzer is None:
        analyzer = TradeAnalyzer()
    return analyzer

def initialize_trade_analysis(initial_portfolio_value: float = 0.0):
    """Initialize the trade analysis system"""
    global analyzer
    analyzer = TradeAnalyzer()
    analyzer.initial_portfolio_value = initial_portfolio_value
    analyzer.current_portfolio_value = initial_portfolio_value
    print(f"Trade analysis initialized for {analyzer.target_trades} trades")
    return analyzer