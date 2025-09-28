#!/usr/bin/env python3
"""
Real-time Trading Bot Monitoring Dashboard
Provides comprehensive monitoring and analytics for trading operations
"""

import json
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
import os
from alert_dashboard import init_alert_dashboard

@dataclass
class TradeMetrics:
    """Real-time trade metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    daily_pnl: float = 0.0
    hourly_pnl: float = 0.0
    trades_per_hour: float = 0.0
    avg_profit_per_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    risk_score: float = 0.0
    volatility: float = 0.0

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    api_response_time: float = 0.0
    blockchain_latency: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0
    active_positions: int = 0
    pending_orders: int = 0
    api_calls_per_minute: int = 0
    gas_price_gwei: float = 0.0
    network_status: str = "Unknown"
    last_update: str = ""

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for trading bot"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.trade_metrics = TradeMetrics()
        self.system_metrics = SystemMetrics()
        self.recent_trades = deque(maxlen=100)
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.pnl_history = deque(maxlen=1000)
        self.error_log = deque(maxlen=500)
        self.performance_alerts = deque(maxlen=50)
        self.risk_alerts = deque(maxlen=50)
        self.critical_risk_start_time: Optional[datetime] = None
        self.critical_risk_alert_triggered: bool = False
        
        # Real-time data tracking
        self.hourly_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'volume': 0.0})
        self.daily_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'volume': 0.0})
        self.symbol_performance = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'avg_duration': 0.0
        })
        
        # Monitoring thresholds
        self.thresholds = {
            'max_daily_loss': -500.0,
            'max_drawdown': -1000.0,
            'min_win_rate': 0.4,
            'max_consecutive_losses': 5,
            'max_risk_score': 0.8,
            'max_error_rate': 0.1,
            'critical_risk_duration_min': 10,
        }
        
        self.start_time = datetime.now()
        self.monitoring_active = True
        self._init_database()
        
    def _init_database(self):
        """Initialize monitoring database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create monitoring tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS monitoring_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,
                        metric_data TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)

                # Add new columns if they don't exist for backward compatibility
                table_info = cursor.execute("PRAGMA table_info(performance_alerts)").fetchall()
                column_names = [info[1] for info in table_info]
                
                if 'resolved_timestamp' not in column_names:
                    cursor.execute("ALTER TABLE performance_alerts ADD COLUMN resolved_timestamp DATETIME")
                
                if 'resolution_time_seconds' not in column_names:
                    cursor.execute("ALTER TABLE performance_alerts ADD COLUMN resolution_time_seconds INTEGER")
                
                conn.commit()
        except Exception as e:
            print(f"Error initializing monitoring database: {e}") # Keep print here as logger may not be available
    
    def update_trade_metrics(self, trade_data: Dict[str, Any]):
        """Update trade metrics with new trade data"""
        try:
            symbol = trade_data.get('symbol', 'UNKNOWN')
            pnl = float(trade_data.get('pnl', 0.0))
            success = trade_data.get('success', False)
            duration = float(trade_data.get('duration', 0.0))
            
            # Update basic metrics
            self.trade_metrics.total_trades += 1
            if success:
                self.trade_metrics.successful_trades += 1
                if pnl > 0:
                    self.trade_metrics.consecutive_wins += 1
                    self.trade_metrics.consecutive_losses = 0
                    if pnl > self.trade_metrics.largest_win:
                        self.trade_metrics.largest_win = pnl
                else:
                    self.trade_metrics.consecutive_wins = 0
                    self.trade_metrics.consecutive_losses += 1
                    if pnl < self.trade_metrics.largest_loss:
                        self.trade_metrics.largest_loss = pnl
            else:
                self.trade_metrics.failed_trades += 1
                self.trade_metrics.consecutive_wins = 0
                self.trade_metrics.consecutive_losses += 1
            
            # Update PnL and balance
            self.trade_metrics.total_pnl += pnl
            self.trade_metrics.current_balance += pnl
            
            if self.trade_metrics.current_balance > self.trade_metrics.peak_balance:
                self.trade_metrics.peak_balance = self.trade_metrics.current_balance
            
            # Calculate drawdown
            # Calculate drawdown as a decimal percentage for consistency
            if self.trade_metrics.peak_balance > 0:
                drawdown_pct = (self.trade_metrics.current_balance - self.trade_metrics.peak_balance) / self.trade_metrics.peak_balance
            else:
                drawdown_pct = 0.0
            
            self.trade_metrics.current_drawdown = drawdown_pct  # e.g., -0.05 for -5%
            if drawdown_pct < self.trade_metrics.max_drawdown:
                self.trade_metrics.max_drawdown = drawdown_pct

            # Update win rate
            if self.trade_metrics.total_trades > 0:
                self.trade_metrics.win_rate = self.trade_metrics.successful_trades / self.trade_metrics.total_trades
            
            # Update symbol performance
            symbol_stats = self.symbol_performance[symbol]
            symbol_stats['trades'] += 1
            symbol_stats['pnl'] += pnl
            if success and pnl > 0:
                symbol_stats['wins'] += 1
            elif success and pnl <= 0:
                symbol_stats['losses'] += 1
            
            # Add to recent trades
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'pnl': pnl,
                'success': success,
                'duration': duration
            }
            self.recent_trades.append(trade_record)
            
            # Update time-based stats
            current_hour = datetime.now().strftime('%Y-%m-%d %H')
            current_day = datetime.now().strftime('%Y-%m-%d')
            
            self.hourly_stats[current_hour]['trades'] += 1
            self.hourly_stats[current_hour]['pnl'] += pnl
            
            self.daily_stats[current_day]['trades'] += 1
            self.daily_stats[current_day]['pnl'] += pnl
            
            # Check for alerts
            self._check_performance_alerts()
            
        except Exception as e:
            self.log_error(f"Error updating trade metrics: {e}")
    
    def update_system_metrics(self, system_data: Dict[str, Any]):
        """Update system performance metrics"""
        try:
            for key, value in system_data.items():
                if hasattr(self.system_metrics, key):
                    setattr(self.system_metrics, key, value)
            
            self.system_metrics.last_update = datetime.now().isoformat()
            self.system_metrics.uptime = (datetime.now() - self.start_time).total_seconds() / 3600

            # Check for risk alerts based on data passed in system_data
            risk_level = system_data.get('risk_level')
            if risk_level:
                self._check_risk_alerts(risk_level)
            
        except Exception as e:
            self.log_error(f"Error updating system metrics: {e}")
    
    def _check_risk_alerts(self, risk_level: str):
        """Check for risk-related alerts, like prolonged critical state."""
        # Check for prolonged CRITICAL risk level
        if risk_level == 'CRITICAL':
            if self.critical_risk_start_time is None:
                # Start the timer when risk becomes critical
                self.critical_risk_start_time = datetime.now()
            
            duration_minutes = (datetime.now() - self.critical_risk_start_time).total_seconds() / 60
            
            # If duration exceeds threshold and we haven't already sent an alert for this period
            if duration_minutes >= self.thresholds['critical_risk_duration_min'] and not self.critical_risk_alert_triggered:
                alert = {
                    'type': 'PROLONGED_CRITICAL_RISK',
                    'severity': 'CRITICAL',
                    'message': f"Risk level has been CRITICAL for over {self.thresholds.get('critical_risk_duration_min', 10)} minutes."
                }
                alert['timestamp'] = datetime.now().isoformat()
                alert_id = self._save_alert_to_db(alert)
                if alert_id:
                    alert['id'] = alert_id
                    self.risk_alerts.append(alert)
                self.critical_risk_alert_triggered = True # Trigger only once per critical period
        else:
            # Reset timer and trigger flag if risk level is no longer critical
            self.critical_risk_start_time = None
            self.critical_risk_alert_triggered = False

    def log_error(self, error_message: str, error_type: str = "GENERAL"):
        """Log error with timestamp"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message
        }
        self.error_log.append(error_record)
        
        # Update error rate
        recent_errors = [e for e in self.error_log if 
                        datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)]
        self.system_metrics.error_rate = len(recent_errors) / 60  # errors per minute
    
    def log_error_alert(self, alert_data: Dict[str, Any]):
        """
        Logs a detailed error alert, saves it to the database, and adds it to the
        in-memory risk alert queue for real-time display.
        """
        alert = {
            'type': alert_data.get('error_type', 'SYSTEM_ERROR'),
            'severity': alert_data.get('severity', 'MEDIUM'),
            'message': alert_data.get('message', 'An unspecified error occurred.')
        }
        alert['timestamp'] = datetime.now().isoformat()
        alert_id = self._save_alert_to_db(alert)
        if alert_id:
            alert['id'] = alert_id
            self.risk_alerts.append(alert) # Use risk_alerts for system/error alerts

    def _check_performance_alerts(self):
        """Check for performance alerts based on thresholds"""
        alerts = []
        
        # Check daily loss
        current_day = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_stats[current_day]['pnl']
        if daily_pnl < self.thresholds['max_daily_loss']:
            alerts.append({
                'type': 'DAILY_LOSS',
                'severity': 'HIGH',
                'message': f"Daily loss exceeded threshold: ${daily_pnl:.2f}"
            })
        
        # Check drawdown
        if self.trade_metrics.current_drawdown < self.thresholds['max_drawdown']:
            alerts.append({
                'type': 'DRAWDOWN',
                'severity': 'HIGH',
                'message': f"Drawdown exceeded threshold: ${self.trade_metrics.current_drawdown:.2f}"
            })
        
        # Check win rate
        if (self.trade_metrics.total_trades > 10 and 
            self.trade_metrics.win_rate < self.thresholds['min_win_rate']):
            alerts.append({
                'type': 'WIN_RATE',
                'severity': 'MEDIUM',
                'message': f"Win rate below threshold: {self.trade_metrics.win_rate:.2%}"
            })
        
        # Check consecutive losses
        if self.trade_metrics.consecutive_losses >= self.thresholds['max_consecutive_losses']:
            alerts.append({
                'type': 'CONSECUTIVE_LOSSES',
                'severity': 'HIGH',
                'message': f"Consecutive losses: {self.trade_metrics.consecutive_losses}"
            })
        
        # Add alerts to queue and database
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert_id = self._save_alert_to_db(alert)
            if alert_id:
                alert['id'] = alert_id
                self.performance_alerts.append(alert)
    
    def _save_alert_to_db(self, alert: Dict[str, Any]) -> Optional[int]:
        """Save an alert to the database and return its ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_alerts (alert_type, severity, message)
                    VALUES (?, ?, ?)
                """, (alert['type'], alert['severity'], alert['message']))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            self.log_error(f"Error saving alert to DB: {e}")
            return None

    def resolve_alert(self, alert_id: int) -> bool:
        """Mark an alert as resolved in the database and remove from active queue."""
        try:
            resolved_time = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the creation timestamp to calculate resolution time
                cursor.execute("SELECT timestamp FROM performance_alerts WHERE id = ?", (alert_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"Alert {alert_id} not found in database.")
                    return False
                
                creation_timestamp_str = result[0]
                # The timestamp is stored in ISO format, so we can parse it back
                creation_timestamp = datetime.fromisoformat(creation_timestamp_str)
                
                resolution_seconds = (resolved_time - creation_timestamp).total_seconds()
                
                # Mark as resolved in DB and store resolution time
                cursor.execute("""
                    UPDATE performance_alerts 
                    SET resolved = TRUE, resolved_timestamp = ?, resolution_time_seconds = ?
                    WHERE id = ?
                """, (resolved_time.isoformat(), int(resolution_seconds), alert_id))
                conn.commit()
            
            # Remove from in-memory deques by creating new deques
            self.performance_alerts = deque([a for a in self.performance_alerts if a.get('id') != alert_id], maxlen=self.performance_alerts.maxlen)
            self.risk_alerts = deque([a for a in self.risk_alerts if a.get('id') != alert_id], maxlen=self.risk_alerts.maxlen)
            
            print(f"Alert {alert_id} marked as resolved in {resolution_seconds:.1f} seconds.")
            return True
        except Exception as e:
            self.log_error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def _calculate_avg_resolution_time(self) -> float:
        """Calculate the average alert resolution time from the database for the last 7 days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("""
                    SELECT AVG(resolution_time_seconds) 
                    FROM performance_alerts 
                    WHERE resolved = TRUE AND resolved_timestamp >= ?
                """, (seven_days_ago,))
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 0.0
        except Exception as e:
            self.log_error(f"Error calculating avg resolution time: {e}")
            return 0.0

    def get_historical_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical alerts from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Makes it easy to convert to dict
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, alert_type, severity, message, resolved, resolved_timestamp, resolution_time_seconds
                    FROM performance_alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                alerts = [dict(row) for row in cursor.fetchall()]
                return alerts
        except Exception as e:
            self.log_error(f"Error retrieving historical alerts: {e}")
            return []

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades from the monitoring dashboard"""
        return list(self.recent_trades)[-limit:]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        avg_resolution_time = self._calculate_avg_resolution_time()
        
        trade_metrics_dict = asdict(self.trade_metrics)
        trade_metrics_dict['avg_alert_resolution_time_sec'] = avg_resolution_time

        return {
            'trade_metrics': trade_metrics_dict,
            'system_metrics': asdict(self.system_metrics),
            'recent_trades': list(self.recent_trades)[-20:],
            'hourly_stats': dict(list(self.hourly_stats.items())[-24:]),
            'daily_stats': dict(list(self.daily_stats.items())[-7:]),
            'symbol_performance': dict(self.symbol_performance),
            'recent_errors': list(self.error_log)[-10:],
            'performance_alerts': list(self.performance_alerts)[-10:],
            'risk_alerts': list(self.risk_alerts)[-10:],
            'thresholds': self.thresholds
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            data = self.get_dashboard_data()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            self.log_error(f"Error exporting metrics: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all metrics (use with caution)"""
        self.trade_metrics = TradeMetrics()
        self.system_metrics = SystemMetrics()
        self.recent_trades.clear()
        self.hourly_stats.clear()
        self.daily_stats.clear()
        self.symbol_performance.clear()
        self.error_log.clear()
        self.performance_alerts.clear()
        self.risk_alerts.clear()
        self.start_time = datetime.now()

# Global monitoring instance
monitoring_dashboard = MonitoringDashboard()

def get_monitoring_dashboard():
    """Get the global monitoring dashboard instance"""
    return monitoring_dashboard

def initialize_dashboard(config=None):
    """Initialize the monitoring dashboard"""
    global monitoring_dashboard
    monitoring_dashboard = MonitoringDashboard()
    return monitoring_dashboard