#!/usr/bin/env python3
"""
Real-time Performance Monitoring System
Tracks KPIs, win rates, and trading performance metrics
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics

@dataclass
class TradeMetrics:
    """Individual trade performance metrics"""
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    duration: Optional[timedelta]
    win: Optional[bool]
    confidence: float
    strategy: str

@dataclass
class PerformanceKPIs:
    """Key Performance Indicators"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    portfolio_value: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    trades_per_hour: float
    avg_trade_duration: float
    risk_reward_ratio: float

class PerformanceMonitor:
    """Real-time performance monitoring and optimization system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trades_history: deque = deque(maxlen=10000)
        self.kpi_history: deque = deque(maxlen=1000)
        self.active_trades: Dict[str, TradeMetrics] = {}
        self.performance_alerts: List[Dict] = []
        
        # Performance targets
        self.target_win_rate = config.get('target_win_rate', 0.65)
        self.target_profit_factor = config.get('target_profit_factor', 1.5)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.15)
        
        # Real-time tracking
        self.current_portfolio_value = config.get('initial_portfolio_value', 10000)
        self.peak_portfolio_value = self.current_portfolio_value
        self.start_time = datetime.now()
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"Performance Monitor initialized - Target Win Rate: {self.target_win_rate:.1%}")
    
    def record_trade_entry(self, symbol: str, side: str, price: float, quantity: float, 
                          confidence: float, strategy: str) -> str:
        """Record a new trade entry"""
        trade_id = f"{symbol}_{side}_{int(time.time())}"
        
        trade = TradeMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            pnl=None,
            pnl_percentage=None,
            duration=None,
            win=None,
            confidence=confidence,
            strategy=strategy
        )
        
        self.active_trades[trade_id] = trade
        print(f"Trade Entry Recorded: {symbol} {side} @ {price} (Confidence: {confidence:.2f})")
        return trade_id
    
    def record_trade_exit(self, trade_id: str, exit_price: float) -> Optional[TradeMetrics]:
        """Record trade exit and calculate performance"""
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        trade.exit_price = exit_price
        trade.duration = datetime.now() - trade.timestamp
        
        # Calculate PnL
        if trade.side == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        trade.win = trade.pnl > 0
        
        # Update portfolio value
        self.current_portfolio_value += trade.pnl
        if self.current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.current_portfolio_value
        
        # Move to history
        self.trades_history.append(trade)
        del self.active_trades[trade_id]
        
        print(f"Trade Exit: {trade.symbol} PnL: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)")
        return trade
    
    def calculate_current_kpis(self) -> PerformanceKPIs:
        """Calculate current performance KPIs"""
        if not self.trades_history:
            return self._get_default_kpis()
        
        trades = list(self.trades_history)
        completed_trades = [t for t in trades if t.pnl is not None]
        
        if not completed_trades:
            return self._get_default_kpis()
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.win])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in completed_trades)
        total_pnl_percentage = (total_pnl / self.config.get('initial_portfolio_value', 10000)) * 100
        
        wins = [t.pnl for t in completed_trades if t.win]
        losses = [abs(t.pnl) for t in completed_trades if not t.win]
        
        avg_win = statistics.mean(wins) if wins else 0
        avg_loss = statistics.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')
        
        # Risk metrics
        current_drawdown = (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value
        max_drawdown = self._calculate_max_drawdown(completed_trades)
        
        # Time-based metrics
        runtime = datetime.now() - self.start_time
        trades_per_hour = total_trades / (runtime.total_seconds() / 3600) if runtime.total_seconds() > 0 else 0
        
        durations = [t.duration.total_seconds() for t in completed_trades if t.duration]
        avg_trade_duration = statistics.mean(durations) if durations else 0
        
        # Returns
        daily_return = self._calculate_period_return(timedelta(days=1))
        weekly_return = self._calculate_period_return(timedelta(weeks=1))
        monthly_return = self._calculate_period_return(timedelta(days=30))
        
        # Risk-reward ratio
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_percentage for t in completed_trades]
        sharpe_ratio = statistics.mean(returns) / statistics.stdev(returns) if len(returns) > 1 else 0
        
        return PerformanceKPIs(
            timestamp=datetime.now(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            portfolio_value=self.current_portfolio_value,
            daily_return=daily_return,
            weekly_return=weekly_return,
            monthly_return=monthly_return,
            trades_per_hour=trades_per_hour,
            avg_trade_duration=avg_trade_duration,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def check_performance_alerts(self, kpis: PerformanceKPIs) -> List[Dict]:
        """Check for performance alerts and optimization triggers"""
        alerts = []
        
        # Win rate alert
        if kpis.win_rate < self.target_win_rate and kpis.total_trades >= 10:
            alerts.append({
                'type': 'WIN_RATE_LOW',
                'message': f'Win rate {kpis.win_rate:.1%} below target {self.target_win_rate:.1%}',
                'severity': 'HIGH',
                'recommendation': 'Adjust entry criteria and risk management'
            })
        
        # Profit factor alert
        if kpis.profit_factor < self.target_profit_factor and kpis.total_trades >= 10:
            alerts.append({
                'type': 'PROFIT_FACTOR_LOW',
                'message': f'Profit factor {kpis.profit_factor:.2f} below target {self.target_profit_factor:.2f}',
                'severity': 'HIGH',
                'recommendation': 'Optimize position sizing and exit strategies'
            })
        
        # Drawdown alert
        if kpis.current_drawdown > self.max_drawdown_threshold:
            alerts.append({
                'type': 'HIGH_DRAWDOWN',
                'message': f'Current drawdown {kpis.current_drawdown:.1%} exceeds threshold',
                'severity': 'CRITICAL',
                'recommendation': 'Reduce position sizes and review risk management'
            })
        
        # Consecutive losses
        recent_trades = list(self.trades_history)[-10:]
        if len(recent_trades) >= 5:
            recent_losses = sum(1 for t in recent_trades if not t.win)
            if recent_losses >= 5:
                alerts.append({
                    'type': 'CONSECUTIVE_LOSSES',
                    'message': f'{recent_losses} consecutive losses detected',
                    'severity': 'HIGH',
                    'recommendation': 'Review strategy parameters and market conditions'
                })
        
        return alerts
    
    def get_optimization_recommendations(self, kpis: PerformanceKPIs) -> List[str]:
        """Generate optimization recommendations based on performance"""
        recommendations = []
        
        if kpis.win_rate < 0.5:
            recommendations.append("Increase confidence threshold for trade entries")
            recommendations.append("Review and optimize technical indicators")
        
        if kpis.avg_loss > kpis.avg_win * 2:
            recommendations.append("Implement tighter stop-loss levels")
            recommendations.append("Consider position sizing optimization")
        
        if kpis.trades_per_hour > 5:
            recommendations.append("Reduce trading frequency to avoid overtrading")
        
        if kpis.sharpe_ratio < 1.0:
            recommendations.append("Improve risk-adjusted returns through better timing")
        
        return recommendations
    
    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        kpis = self.calculate_current_kpis()
        alerts = self.check_performance_alerts(kpis)
        recommendations = self.get_optimization_recommendations(kpis)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'kpis': asdict(kpis),
            'alerts': alerts,
            'recommendations': recommendations,
            'active_trades': len(self.active_trades),
            'total_trades_history': len(self.trades_history),
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                kpis = self.calculate_current_kpis()
                self.kpi_history.append(kpis)
                
                # Check for alerts
                alerts = self.check_performance_alerts(kpis)
                if alerts:
                    self.performance_alerts.extend(alerts)
                    for alert in alerts:
                        print(f"PERFORMANCE ALERT [{alert['severity']}]: {alert['message']}")
                
                # Log performance summary every 10 minutes
                if len(self.kpi_history) % 10 == 0:
                    print(f"Performance Summary - Win Rate: {kpis.win_rate:.1%}, "
                          f"PnL: ${kpis.total_pnl:.2f}, Trades: {kpis.total_trades}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(60)
    
    def _get_default_kpis(self) -> PerformanceKPIs:
        """Get default KPIs when no trades exist"""
        return PerformanceKPIs(
            timestamp=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_percentage=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            portfolio_value=self.current_portfolio_value,
            daily_return=0.0,
            weekly_return=0.0,
            monthly_return=0.0,
            trades_per_hour=0.0,
            avg_trade_duration=0.0,
            risk_reward_ratio=0.0
        )
    
    def _calculate_max_drawdown(self, trades: List[TradeMetrics]) -> float:
        """Calculate maximum drawdown from trade history"""
        if not trades:
            return 0.0
        
        portfolio_values = []
        running_value = self.config.get('initial_portfolio_value', 10000)
        
        for trade in trades:
            running_value += trade.pnl
            portfolio_values.append(running_value)
        
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_period_return(self, period: timedelta) -> float:
        """Calculate return for a specific period"""
        cutoff_time = datetime.now() - period
        period_trades = [t for t in self.trades_history if t.timestamp >= cutoff_time]
        
        if not period_trades:
            return 0.0
        
        period_pnl = sum(t.pnl for t in period_trades)
        return (period_pnl / self.config.get('initial_portfolio_value', 10000)) * 100
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("Performance monitoring stopped")

if __name__ == "__main__":
    # Test the performance monitor
    config = {
        'target_win_rate': 0.65,
        'target_profit_factor': 1.5,
        'max_drawdown_threshold': 0.15,
        'initial_portfolio_value': 10000
    }
    
    monitor = PerformanceMonitor(config)
    
    # Simulate some trades
    trade_id1 = monitor.record_trade_entry('BTCUSDT', 'BUY', 50000, 0.1, 0.8, 'ML_STRATEGY')
    time.sleep(1)
    monitor.record_trade_exit(trade_id1, 51000)
    
    trade_id2 = monitor.record_trade_entry('ETHUSDT', 'BUY', 3000, 1.0, 0.7, 'TECHNICAL_STRATEGY')
    time.sleep(1)
    monitor.record_trade_exit(trade_id2, 2950)
    
    # Get performance report
    report = monitor.export_performance_report()
    print(json.dumps(report, indent=2, default=str))
    
    time.sleep(5)
    monitor.stop_monitoring()