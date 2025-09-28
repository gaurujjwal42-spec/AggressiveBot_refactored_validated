# -*- coding: utf-8 -*-
"""
Enhanced Logging System for Trading Bot

Provides comprehensive logging capabilities including:
- Trade execution tracking
- Performance metrics logging
- Risk assessment logging
- Real-time monitoring
- File-based persistent logging
"""

import os
import json
import csv
import logging
import traceback
import time
import functools
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import threading

# Import local modules for integration
from logging.handlers import TimedRotatingFileHandler

class EnhancedLogger:
    """Enhanced logging system for trading operations."""
    
    def __init__(self, log_dir: str = "logs", max_log_files: int = 30):
        self.log_dir = log_dir
        self.max_log_files = max_log_files
        self.trade_logs = deque(maxlen=1000)
        self.performance_logs = deque(maxlen=500)
        self.risk_logs = deque(maxlen=500)
        self.error_logs = deque(maxlen=200)
        self.api_logs = deque(maxlen=300)  # New API request/response logs
        self.system_logs = deque(maxlen=300)  # New system event logs
        self.lock = threading.Lock()
        
        # Create logs directory and subdirectories
        for subdir in ['', 'api', 'system', 'errors', 'performance', 'trades', 'risk']:
            os.makedirs(os.path.join(log_dir, subdir) if subdir else log_dir, exist_ok=True)
        
        # Setup file loggers
        self._setup_loggers()
        
        # Cleanup old log files
        self._cleanup_old_logs()
        
        # Initialize monitoring dashboard integration
        self._init_monitoring_dashboard()
        
    def _init_monitoring_dashboard(self):
        """Initialize connection to monitoring dashboard"""
        try:
            # Moved import here to break circular dependency
            from monitoring_dashboard import get_monitoring_dashboard
            self.monitoring_dashboard = get_monitoring_dashboard()
        except Exception as e:
            print(f"Warning: Could not initialize monitoring dashboard: {e}")
            self.monitoring_dashboard = None
    
    def _setup_loggers(self):
        """Setup file-based loggers for different log types."""
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        def setup_rotating_logger(name: str, level: int, subdir: str, filename_prefix: str) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            # Prevent adding handlers multiple times if this is called again
            if logger.hasHandlers():
                logger.handlers.clear()
            
            log_path = os.path.join(self.log_dir, subdir, f"{filename_prefix}.log")
            handler = TimedRotatingFileHandler(
                log_path,
                when="midnight",
                interval=1,
                backupCount=self.max_log_files
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger

        self.trade_logger = setup_rotating_logger('trades', logging.INFO, 'trades', 'trades')
        self.performance_logger = setup_rotating_logger('performance', logging.INFO, 'performance', 'performance')
        self.risk_logger = setup_rotating_logger('risk', logging.INFO, 'risk', 'risk')
        self.error_logger = setup_rotating_logger('errors', logging.ERROR, 'errors', 'errors')
        self.api_logger = setup_rotating_logger('api', logging.INFO, 'api', 'api')
        self.system_logger = setup_rotating_logger('system', logging.INFO, 'system', 'system')
        self.debug_logger = setup_rotating_logger('debug', logging.DEBUG, '', 'debug')
    
    def _cleanup_old_logs(self):
        """Remove log files older than max_log_files days."""
        try:
            log_subdirs = [d for d in os.listdir(self.log_dir) if os.path.isdir(os.path.join(self.log_dir, d))]
            log_subdirs.append('') # Also check the root log directory
            cutoff_date = datetime.now() - timedelta(days=self.max_log_files)
            for subdir in log_subdirs:
                current_dir = os.path.join(self.log_dir, subdir)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.log') or filename.endswith('.csv'):
                        file_path = os.path.join(current_dir, filename)
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up old logs: {e}")
    
    def log_trade_execution(self, trade_data: Dict[str, Any]):
        """Log trade execution details with enhanced information."""
        with self.lock:
            # Ensure timestamp exists
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
                
            # Add execution context if not present
            if 'execution_context' not in trade_data:
                trade_data['execution_context'] = {
                    'logged_at': datetime.now().isoformat(),
                    'execution_id': f"trade_{int(time.time()*1000)}"
                }
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'TRADE_EXECUTION',
                'data': trade_data
            }
            
            self.trade_logs.append(log_entry)
            
            # Log to file
            self.trade_logger.info(json.dumps(log_entry))
            
            # Also log to CSV for easy analysis
            self._log_trade_to_csv(trade_data)
            
            # Update monitoring dashboard if available
            if hasattr(self, 'monitoring_dashboard') and self.monitoring_dashboard:
                try:
                    self.monitoring_dashboard.update_trade_metrics(trade_data)
                except Exception as e:
                    self.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        with self.lock:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'PERFORMANCE_METRICS',
                'data': metrics
            }
            
            self.performance_logs.append(log_entry)
            
            # Log to file
            self.performance_logger.info(json.dumps(log_entry))
    
    def log_risk_assessment(self, risk_data: Dict[str, Any]):
        """Log risk assessment details."""
        with self.lock:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'RISK_ASSESSMENT',
                'data': risk_data
            }
            
            self.risk_logs.append(log_entry)
            
            # Log to file
            self.risk_logger.info(json.dumps(log_entry))
    
    def _log_trade_to_csv(self, trade_data: Dict[str, Any]):
        """Log trade details to a CSV file for easy analysis"""
        csv_file = os.path.join(self.log_dir, f'trade_history_{datetime.now().strftime("%Y%m%d")}.csv')
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'symbol', 'type', 'price', 'usdt_amount', 
                'token_amount', 'pnl', 'reason', 'order_id', 'status', 
                'execution_time_ms', 'success'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Prepare row data
            row = {
                'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
                'symbol': trade_data.get('symbol', ''),
                'type': trade_data.get('type', ''),
                'price': trade_data.get('price', 0),
                'usdt_amount': trade_data.get('usdt_amount', 0),
                'token_amount': trade_data.get('token_amount', 0),
                'pnl': trade_data.get('pnl', 0),
                'reason': trade_data.get('reason', ''),
                'order_id': trade_data.get('id', ''),
                'status': trade_data.get('status', ''),
                'execution_time_ms': trade_data.get('execution_time_ms', 0),
                'success': trade_data.get('success', False)
            }
            
            writer.writerow(row)
    
    def log_info(self, info_msg: str, details: Optional[Dict] = None):
        """Log informational message."""
        with self.lock:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'INFO',
                'message': info_msg,
                'details': details or {}
            }
            
            # Log to file
            self.system_logger.info(json.dumps(log_entry))
    
    def log_error(self, error_msg: str, error_type: str = "GENERAL", details: Optional[Dict] = None):
        """Log error with details and stack trace."""
        with self.lock:
            # Get current stack trace
            stack_trace = traceback.format_stack()
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'ERROR',
                'error_type': error_type,
                'message': error_msg,
                'details': details or {},
                'stack_trace': stack_trace
            }
            
            self.error_logs.append(log_entry)
            
            # Log to file
            self.error_logger.error(json.dumps(log_entry))
            
            # Update monitoring dashboard with error if available
            if hasattr(self, 'monitoring_dashboard') and self.monitoring_dashboard:
                try:
                    self.monitoring_dashboard.log_error({
                        'timestamp': datetime.now().isoformat(),
                        'error_type': error_type,
                        'message': error_msg,
                        'context': details
                    })
                except Exception as e:
                    # Don't try to update dashboard again to avoid recursion
                    self.error_logger.error(f"Failed to update monitoring dashboard with error: {str(e)}")
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade logs."""
        with self.lock:
            return list(self.trade_logs)[-limit:]
    
    def get_recent_performance(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent performance logs."""
        with self.lock:
            return list(self.performance_logs)[-limit:]
    
    def get_recent_risks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent risk assessment logs."""
        with self.lock:
            return list(self.risk_logs)[-limit:]
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error logs."""
        with self.lock:
            return list(self.error_logs)[-limit:]
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading summary."""
        with self.lock:
            recent_trades = list(self.trade_logs)[-100:]  # Last 100 trades
            
            if not recent_trades:
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'success_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0
                }
            
            successful_trades = 0
            failed_trades = 0
            total_pnl = 0.0
            
            for trade in recent_trades:
                trade_data = trade.get('data', {})
                if trade_data.get('success', False):
                    successful_trades += 1
                    total_pnl += trade_data.get('pnl', 0)
                else:
                    failed_trades += 1
            
            total_trades = successful_trades + failed_trades
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = total_pnl / successful_trades if successful_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'success_rate': success_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': avg_pnl
            }
    
    def log_api_request(self, endpoint: str, method: str, params: Dict[str, Any]):
        """Log API request details"""
        with self.lock:
            request_data = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'method': method,
                'params': params
            }
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'API_REQUEST',
                'data': request_data
            }
            
            self.api_logs.append(log_entry)
            
            # Log to file
            self.api_logger.info(json.dumps(log_entry))
    
    def log_api_response(self, endpoint: str, method: str, response: Dict[str, Any], execution_time: float):
        """Log API response details"""
        with self.lock:
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'method': method,
                'response': response,
                'execution_time_ms': execution_time
            }
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'API_RESPONSE',
                'data': response_data
            }
            
            self.api_logs.append(log_entry)
            
            # Log to file
            self.api_logger.info(json.dumps(log_entry))
            
            # Update monitoring dashboard with API response time if available
            if hasattr(self, 'monitoring_dashboard') and self.monitoring_dashboard:
                try:
                    self.monitoring_dashboard.system_metrics.api_response_time = execution_time
                except Exception as e:
                    self.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events"""
        with self.lock:
            event_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': details
            }
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'SYSTEM_EVENT',
                'data': event_data
            }
            
            self.system_logs.append(log_entry)
            
            # Log to file
            self.system_logger.info(json.dumps(log_entry))
            
            # Log to JSON for detailed analysis
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"system_event_{timestamp_str}.json"
            json_file_path = os.path.join(self.log_dir, 'system', json_filename)
            
            with open(json_file_path, 'w') as json_file:
                json.dump(event_data, json_file, indent=4)
    
    def log_debug(self, message: str, data: Dict[str, Any] = None):
        """Log debug information for troubleshooting"""
        if data is None:
            data = {}
            
        with self.lock:
            debug_data = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'data': data
            }
            
            # Log to debug logger
            self.debug_logger.debug(f"{message} - {json.dumps(data)}")
    
    def export_logs_to_json(self, filename: Optional[str] = None) -> str:
        """Export all logs to a JSON file."""
        if not filename:
            filename = f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with self.lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'trade_logs': list(self.trade_logs),
                'performance_logs': list(self.performance_logs),
                'risk_logs': list(self.risk_logs),
                'error_logs': list(self.error_logs),
                'api_logs': list(self.api_logs),
                'system_logs': list(self.system_logs)
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
        
    def get_recent_api_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent API request/response logs."""
        with self.lock:
            return list(self.api_logs)[-limit:]
    
    def get_recent_system_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent system event logs."""
        with self.lock:
            return list(self.system_logs)[-limit:]
            
    def track_execution_time(self, function_name: str = None):
        """Decorator to track execution time of functions"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    # Log performance metric
                    func_name = function_name or func.__name__
                    self.log_performance_metrics({
                        'function': func_name,
                        'execution_time_ms': execution_time,
                        'success': True
                    })
                    
                    # Update monitoring dashboard if available
                    if hasattr(self, 'monitoring_dashboard') and self.monitoring_dashboard:
                        try:
                            # Update function-specific metrics
                            if 'trade' in func_name.lower() or 'order' in func_name.lower():
                                self.monitoring_dashboard.system_metrics.trade_execution_time = execution_time
                            elif 'api' in func_name.lower() or 'request' in func_name.lower():
                                self.monitoring_dashboard.system_metrics.api_response_time = execution_time
                        except Exception as e:
                            self.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
                    
                    return result
                except Exception as e:
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    # Log performance metric with error
                    func_name = function_name or func.__name__
                    self.log_performance_metrics({
                        'function': func_name,
                        'execution_time_ms': execution_time,
                        'success': False,
                        'error': str(e)
                    })
                    
                    # Log the error
                    self.log_error(
                        f"Error in {func_name}: {str(e)}",
                        "FUNCTION_ERROR",
                        {
                            'function': func_name,
                            'args': str(args),
                            'kwargs': str(kwargs),
                            'execution_time_ms': execution_time
                        }
                    )
                    
                    # Re-raise the exception
                    raise
            return wrapper
        return decorator

# Create TradingLogger alias for backward compatibility
TradingLogger = EnhancedLogger