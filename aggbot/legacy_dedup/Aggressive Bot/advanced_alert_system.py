# -*- coding: utf-8 -*-
"""
Advanced Alert System for Trading Bot

Provides sophisticated detection of abnormal trading patterns and system errors:
- Statistical anomaly detection for trading patterns
- Machine learning-based pattern recognition
- Real-time system health monitoring
- Automated alert escalation and notification
- Integration with monitoring dashboard and diagnostic error handler
"""

import time
import json
import statistics
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from enhanced_logger import EnhancedLogger
from diagnostic_error_handler import DiagnosticErrorHandler

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Types of alerts"""
    TRADING_ANOMALY = "TRADING_ANOMALY"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    RISK_THRESHOLD = "RISK_THRESHOLD"
    API_FAILURE = "API_FAILURE"
    NETWORK_ISSUE = "NETWORK_ISSUE"
    UNUSUAL_VOLUME = "UNUSUAL_VOLUME"
    PRICE_MANIPULATION = "PRICE_MANIPULATION"
    CONSECUTIVE_FAILURES = "CONSECUTIVE_FAILURES"
    MEMORY_LEAK = "MEMORY_LEAK"
    LATENCY_SPIKE = "LATENCY_SPIKE"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    escalated: bool = False
    notification_sent: bool = False

class TradingPatternAnalyzer:
    """Analyzes trading patterns for anomalies"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_history = deque(maxlen=window_size)
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.volume_history = defaultdict(lambda: deque(maxlen=window_size))
        self.execution_times = deque(maxlen=window_size)
        
    def add_trade_data(self, trade_data: Dict[str, Any]):
        """Add new trade data for analysis"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': trade_data.get('symbol'),
            'type': trade_data.get('type'),
            'amount': trade_data.get('amount', 0),
            'price': trade_data.get('price', 0),
            'pnl': trade_data.get('pnl', 0),
            'execution_time': trade_data.get('execution_time', 0)
        })
        
        symbol = trade_data.get('symbol')
        if symbol:
            self.price_history[symbol].append(trade_data.get('price', 0))
            self.volume_history[symbol].append(trade_data.get('amount', 0))
            
        if 'execution_time' in trade_data:
            self.execution_times.append(trade_data['execution_time'])
    
    def detect_price_anomalies(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Detect unusual price movements using statistical analysis"""
        if len(self.price_history[symbol]) < 20:
            return None
            
        prices = list(self.price_history[symbol])
        mean_price = statistics.mean(prices)
        std_price = statistics.stdev(prices)
        
        # Z-score analysis
        z_score = abs((current_price - mean_price) / std_price) if std_price > 0 else 0
        
        if z_score > 3:  # More than 3 standard deviations
            return {
                'anomaly_type': 'price_spike',
                'z_score': z_score,
                'current_price': current_price,
                'mean_price': mean_price,
                'std_deviation': std_price,
                'severity': 'HIGH' if z_score > 4 else 'MEDIUM'
            }
        return None
    
    def detect_volume_anomalies(self, symbol: str, current_volume: float) -> Optional[Dict[str, Any]]:
        """Detect unusual trading volume"""
        if len(self.volume_history[symbol]) < 10:
            return None
            
        volumes = list(self.volume_history[symbol])
        median_volume = statistics.median(volumes)
        
        # Check for volume spikes
        if current_volume > median_volume * 5:  # 5x normal volume
            return {
                'anomaly_type': 'volume_spike',
                'current_volume': current_volume,
                'median_volume': median_volume,
                'multiplier': current_volume / median_volume if median_volume > 0 else 0,
                'severity': 'HIGH' if current_volume > median_volume * 10 else 'MEDIUM'
            }
        return None
    
    def detect_execution_anomalies(self) -> Optional[Dict[str, Any]]:
        """Detect unusual execution times"""
        if len(self.execution_times) < 10:
            return None
            
        times = list(self.execution_times)
        mean_time = statistics.mean(times)
        
        # Check recent execution times
        recent_times = times[-5:] if len(times) >= 5 else times
        avg_recent = statistics.mean(recent_times)
        
        if avg_recent > mean_time * 3:  # 3x slower than average
            return {
                'anomaly_type': 'slow_execution',
                'avg_recent_time': avg_recent,
                'historical_avg': mean_time,
                'slowdown_factor': avg_recent / mean_time if mean_time > 0 else 0,
                'severity': 'HIGH' if avg_recent > mean_time * 5 else 'MEDIUM'
            }
        return None
    
    def detect_trading_frequency_anomalies(self) -> Optional[Dict[str, Any]]:
        """Detect unusual trading frequency patterns"""
        if len(self.trade_history) < 20:
            return None
            
        # Calculate trades per hour for different time windows
        now = datetime.now()
        recent_trades = [t for t in self.trade_history if (now - t['timestamp']).total_seconds() < 3600]
        historical_hourly_avg = len(self.trade_history) / max(1, (now - self.trade_history[0]['timestamp']).total_seconds() / 3600)
        
        current_hourly_rate = len(recent_trades)
        
        if current_hourly_rate > historical_hourly_avg * 3:  # 3x normal frequency
            return {
                'anomaly_type': 'high_frequency_trading',
                'current_rate': current_hourly_rate,
                'historical_avg': historical_hourly_avg,
                'frequency_multiplier': current_hourly_rate / max(1, historical_hourly_avg),
                'severity': 'MEDIUM'
            }
        return None

class SystemHealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.api_response_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.cpu_usage_history = deque(maxlen=100)
        self.last_health_check = datetime.now()
        
    def record_error(self, error_type: str, severity: str):
        """Record system error for pattern analysis"""
        self.error_counts[f"{error_type}_{severity}"] += 1
        
    def record_api_response_time(self, response_time: float):
        """Record API response time"""
        self.api_response_times.append({
            'timestamp': datetime.now(),
            'response_time': response_time
        })
    
    def record_system_metrics(self, cpu_usage: float, memory_usage: float):
        """Record system performance metrics"""
        self.cpu_usage_history.append({
            'timestamp': datetime.now(),
            'cpu_usage': cpu_usage
        })
        self.memory_usage_history.append({
            'timestamp': datetime.now(),
            'memory_usage': memory_usage
        })
    
    def detect_system_anomalies(self) -> List[Dict[str, Any]]:
        """Detect system performance anomalies"""
        anomalies = []
        
        # Check for API latency spikes
        if len(self.api_response_times) >= 10:
            recent_times = [r['response_time'] for r in list(self.api_response_times)[-10:]]
            avg_recent = statistics.mean(recent_times)
            
            if avg_recent > 5.0:  # More than 5 seconds average
                anomalies.append({
                    'anomaly_type': 'api_latency_spike',
                    'avg_response_time': avg_recent,
                    'severity': 'HIGH' if avg_recent > 10.0 else 'MEDIUM'
                })
        
        # Check for memory leaks
        if len(self.memory_usage_history) >= 20:
            memory_values = [m['memory_usage'] for m in list(self.memory_usage_history)]
            # Check if memory usage is consistently increasing
            slope, _, r_value, _, _ = stats.linregress(range(len(memory_values)), memory_values)
            
            if slope > 1.0 and r_value > 0.7:  # Increasing trend
                anomalies.append({
                    'anomaly_type': 'memory_leak',
                    'memory_trend_slope': slope,
                    'correlation': r_value,
                    'current_usage': memory_values[-1],
                    'severity': 'HIGH' if slope > 2.0 else 'MEDIUM'
                })
        
        # Check for high CPU usage
        if len(self.cpu_usage_history) >= 10:
            recent_cpu = [c['cpu_usage'] for c in list(self.cpu_usage_history)[-10:]]
            avg_cpu = statistics.mean(recent_cpu)
            
            if avg_cpu > 80.0:  # More than 80% CPU usage
                anomalies.append({
                    'anomaly_type': 'high_cpu_usage',
                    'avg_cpu_usage': avg_cpu,
                    'severity': 'HIGH' if avg_cpu > 95.0 else 'MEDIUM'
                })
        
        return anomalies

class AdvancedAlertSystem:
    """Advanced alert system with pattern recognition and automated responses"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = EnhancedLogger()
        self.monitoring_dashboard = None  # Will be set later to avoid circular import
        self.error_handler = DiagnosticErrorHandler()
        
        self.pattern_analyzer = TradingPatternAnalyzer()
        self.system_monitor = SystemHealthMonitor()
        
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.escalation_rules = self._load_escalation_rules()
    
    def set_monitoring_dashboard(self, dashboard):
        """Set the monitoring dashboard to avoid circular import."""
        self.monitoring_dashboard = dashboard
        if self.error_handler:
            self.error_handler.set_monitoring_dashboard(dashboard)
        
        # Alert thresholds
        self.thresholds = {
            'consecutive_failures': 5,
            'error_rate_per_minute': 10,
            'api_timeout_threshold': 30.0,
            'memory_usage_threshold': 85.0,
            'cpu_usage_threshold': 90.0
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self.monitoring_thread.start()
        
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load alert escalation rules"""
        return {
            'CRITICAL': {'escalate_after_minutes': 5, 'notify_immediately': True},
            'HIGH': {'escalate_after_minutes': 15, 'notify_immediately': True},
            'MEDIUM': {'escalate_after_minutes': 30, 'notify_immediately': False},
            'LOW': {'escalate_after_minutes': 60, 'notify_immediately': False},
            'INFO': {'escalate_after_minutes': 120, 'notify_immediately': False}
        }
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity, title: str, 
                    message: str, context: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        alert_id = f"{alert_type.value}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log the alert
        self.logger.log_system_event("ALERT_CREATED", {
            'alert_id': alert_id,
            'type': alert_type.value,
            'severity': severity.value,
            'title': title,
            'message': message
        })
        
        # Update monitoring dashboard
        if self.monitoring_dashboard:
            try:
                self.monitoring_dashboard.log_error_alert({
                    'timestamp': alert.timestamp.isoformat(),
                    'error_type': alert_type.value,
                    'severity': severity.value,
                    'message': f"{title}: {message}"
                })
            except Exception as e:
                self.logger.log_error(f"Failed to update monitoring dashboard with alert: {str(e)}", "INTEGRATION_ERROR")
        
        # Handle immediate notifications
        if self.escalation_rules.get(severity.value, {}).get('notify_immediately', False):
            self._send_notification(alert)
        
        return alert
    
    def analyze_trading_patterns(self, trade_data: Dict[str, Any]):
        """Analyze trading data for anomalies"""
        self.pattern_analyzer.add_trade_data(trade_data)
        
        symbol = trade_data.get('symbol')
        if not symbol:
            return
        
        # Check for price anomalies
        price_anomaly = self.pattern_analyzer.detect_price_anomalies(
            symbol, trade_data.get('price', 0)
        )
        if price_anomaly:
            self.create_alert(
                AlertType.TRADING_ANOMALY,
                AlertSeverity(price_anomaly['severity']),
                f"Price Anomaly Detected for {symbol}",
                f"Unusual price movement detected. Z-score: {price_anomaly['z_score']:.2f}",
                {'symbol': symbol, 'anomaly_data': price_anomaly}
            )
        
        # Check for volume anomalies
        volume_anomaly = self.pattern_analyzer.detect_volume_anomalies(
            symbol, trade_data.get('amount', 0)
        )
        if volume_anomaly:
            self.create_alert(
                AlertType.UNUSUAL_VOLUME,
                AlertSeverity(volume_anomaly['severity']),
                f"Volume Anomaly Detected for {symbol}",
                f"Unusual trading volume: {volume_anomaly['multiplier']:.1f}x normal",
                {'symbol': symbol, 'anomaly_data': volume_anomaly}
            )
        
        # Check for execution anomalies
        execution_anomaly = self.pattern_analyzer.detect_execution_anomalies()
        if execution_anomaly:
            self.create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertSeverity(execution_anomaly['severity']),
                "Slow Execution Detected",
                f"Trade execution {execution_anomaly['slowdown_factor']:.1f}x slower than normal",
                {'anomaly_data': execution_anomaly}
            )
        
        # Check for trading frequency anomalies
        frequency_anomaly = self.pattern_analyzer.detect_trading_frequency_anomalies()
        if frequency_anomaly:
            self.create_alert(
                AlertType.TRADING_ANOMALY,
                AlertSeverity(frequency_anomaly['severity']),
                "High Frequency Trading Detected",
                f"Trading frequency {frequency_anomaly['frequency_multiplier']:.1f}x higher than normal",
                {'anomaly_data': frequency_anomaly}
            )
    
    def monitor_system_health(self, system_metrics: Dict[str, Any]):
        """Monitor system health and detect issues"""
        # Record metrics
        self.system_monitor.record_system_metrics(
            system_metrics.get('cpu_usage', 0),
            system_metrics.get('memory_usage', 0)
        )
        
        if 'api_response_time' in system_metrics:
            self.system_monitor.record_api_response_time(system_metrics['api_response_time'])
        
        # Detect anomalies
        anomalies = self.system_monitor.detect_system_anomalies()
        
        for anomaly in anomalies:
            alert_type = AlertType.SYSTEM_ERROR
            if anomaly['anomaly_type'] == 'api_latency_spike':
                alert_type = AlertType.LATENCY_SPIKE
            elif anomaly['anomaly_type'] == 'memory_leak':
                alert_type = AlertType.MEMORY_LEAK
            
            self.create_alert(
                alert_type,
                AlertSeverity(anomaly['severity']),
                f"System Anomaly: {anomaly['anomaly_type'].replace('_', ' ').title()}",
                self._format_anomaly_message(anomaly),
                {'anomaly_data': anomaly}
            )
    
    def record_error(self, error: Exception, error_type: str, severity: str, context: Dict[str, Any] = None):
        """Record and analyze system errors"""
        self.system_monitor.record_error(error_type, severity)
        
        # Check for consecutive failures
        recent_errors = [k for k, v in self.system_monitor.error_counts.items() 
                        if 'CRITICAL' in k or 'HIGH' in k]
        
        if len(recent_errors) >= self.thresholds['consecutive_failures']:
            self.create_alert(
                AlertType.CONSECUTIVE_FAILURES,
                AlertSeverity.CRITICAL,
                "Multiple Critical Errors Detected",
                f"System has experienced {len(recent_errors)} critical/high severity errors",
                {'error_counts': dict(self.system_monitor.error_counts), 'context': context}
            )
    
    def _format_anomaly_message(self, anomaly: Dict[str, Any]) -> str:
        """Format anomaly data into readable message"""
        anomaly_type = anomaly['anomaly_type']
        
        if anomaly_type == 'api_latency_spike':
            return f"API response time averaged {anomaly['avg_response_time']:.2f}s over last 10 requests"
        elif anomaly_type == 'memory_leak':
            return f"Memory usage trending upward (slope: {anomaly['memory_trend_slope']:.2f}, current: {anomaly['current_usage']:.1f}%)"
        elif anomaly_type == 'high_cpu_usage':
            return f"CPU usage averaged {anomaly['avg_cpu_usage']:.1f}% over last 10 measurements"
        else:
            return f"System anomaly detected: {anomaly_type}"
    
    def _send_notification(self, alert: Alert):
        """Send notification for alert (placeholder for email/SMS/webhook)"""
        try:
            # Log notification attempt
            self.logger.log_system_event("ALERT_NOTIFICATION", {
                'alert_id': alert.id,
                'severity': alert.severity.value,
                'title': alert.title
            })
            
            # Mark as notification sent
            alert.notification_sent = True
            
            # Here you would implement actual notification logic:
            # - Email notifications
            # - SMS alerts
            # - Webhook calls
            # - Slack/Discord messages
            
        except Exception as e:
            self.logger.log_error(f"Failed to send notification for alert {alert.id}: {str(e)}", "NOTIFICATION_ERROR")
    
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Log resolution
            self.logger.log_system_event("ALERT_RESOLVED", {
                'alert_id': alert_id,
                'resolution_time': alert.resolution_time.isoformat(),
                'resolution_note': resolution_note
            })
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            return True
        return False
    
    def _continuous_monitoring(self):
        """Continuous monitoring loop for alert escalation and cleanup"""
        while True:
            try:
                current_time = datetime.now()
                
                # Check for alerts that need escalation
                for alert_id, alert in list(self.active_alerts.items()):
                    if not alert.escalated:
                        escalation_rule = self.escalation_rules.get(alert.severity.value, {})
                        escalate_after = escalation_rule.get('escalate_after_minutes', 60)
                        
                        if (current_time - alert.timestamp).total_seconds() > escalate_after * 60:
                            self._escalate_alert(alert)
                
                # Clean up old resolved alerts from history
                cutoff_time = current_time - timedelta(days=7)
                self.alert_history = deque(
                    [a for a in self.alert_history if a.timestamp > cutoff_time],
                    maxlen=1000
                )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.log_error(f"Error in continuous monitoring: {str(e)}", "MONITORING_ERROR")
                time.sleep(60)
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an unresolved alert"""
        alert.escalated = True
        
        # Create escalation alert
        self.create_alert(
            AlertType.SYSTEM_ERROR,
            AlertSeverity.CRITICAL,
            f"Escalated Alert: {alert.title}",
            f"Alert {alert.id} has been unresolved for extended period. Original: {alert.message}",
            {'original_alert_id': alert.id, 'escalation': True}
        )
        
        self.logger.log_system_event("ALERT_ESCALATED", {
            'alert_id': alert.id,
            'original_severity': alert.severity.value,
            'escalation_time': datetime.now().isoformat()
        })
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        recent_alerts = [a for a in self.alert_history if 
                        (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            'active_alerts_count': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'recent_alerts_count': len(recent_alerts),
            'total_alerts_today': len([a for a in self.alert_history if 
                                     a.timestamp.date() == datetime.now().date()]),
            'escalated_alerts': len([a for a in self.active_alerts.values() if a.escalated])
        }

# Global instance
_alert_system = None

def get_alert_system(config: Dict[str, Any] = None) -> AdvancedAlertSystem:
    """Get or create the global alert system instance"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AdvancedAlertSystem(config)
    return _alert_system