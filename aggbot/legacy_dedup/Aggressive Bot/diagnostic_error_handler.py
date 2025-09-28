# -*- coding: utf-8 -*-
"""
Diagnostic Error Handling System for Trading Bot

Extends the AdvancedErrorHandler with enhanced diagnostic capabilities:
- Detailed error diagnostics with context-aware messages
- Root cause analysis for complex error chains
- User-friendly error messages with suggested actions
- Integration with monitoring dashboard for real-time error tracking
- Error categorization and prioritization
"""

import time
import traceback
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import deque
from enum import Enum
import requests

from error_handler import AdvancedErrorHandler, ErrorType, CircuitBreakerState
from enhanced_logger import EnhancedLogger

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "CRITICAL"  # System cannot function, immediate attention required
    HIGH = "HIGH"          # Major functionality impaired, prompt attention needed
    MEDIUM = "MEDIUM"      # Partial functionality affected, attention needed soon
    LOW = "LOW"            # Minor issue, can be addressed later
    INFO = "INFO"          # Informational only, no immediate action required

class DiagnosticErrorHandler(AdvancedErrorHandler):
    """Enhanced error handler with diagnostic capabilities."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        super().__init__(max_retries, base_delay)
        self.logger = EnhancedLogger()
        self.monitoring_dashboard = None  # Will be set later to avoid circular import
        self.error_patterns = self._load_error_patterns()
        self.recent_diagnostics = deque(maxlen=100)
    
    def set_monitoring_dashboard(self, dashboard):
        """Set the monitoring dashboard to avoid circular import."""
        self.monitoring_dashboard = dashboard
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known error patterns and their diagnostic information."""
        patterns_file = os.path.join(os.path.dirname(__file__), 'error_patterns.json')
        
        # Default patterns if file doesn't exist
        default_patterns = {
            "rate_limit": {
                "keywords": ["rate limit", "429", "too many requests"],
                "severity": "MEDIUM",
                "diagnosis": "API rate limit exceeded",
                "suggested_action": "Implement rate limiting or increase delay between requests"
            },
            "insufficient_funds": {
                "keywords": ["insufficient", "balance", "not enough"],
                "severity": "HIGH",
                "diagnosis": "Insufficient funds for the requested operation",
                "suggested_action": "Add funds to the account or reduce order size"
            },
            "network_error": {
                "keywords": ["network", "connection", "timeout", "unreachable"],
                "severity": "MEDIUM",
                "diagnosis": "Network connectivity issue",
                "suggested_action": "Check internet connection and retry"
            },
            "authentication": {
                "keywords": ["auth", "unauthorized", "permission", "401"],
                "severity": "HIGH",
                "diagnosis": "Authentication or authorization failure",
                "suggested_action": "Verify API keys and permissions"
            },
            "invalid_parameter": {
                "keywords": ["invalid", "parameter", "argument", "400"],
                "severity": "MEDIUM",
                "diagnosis": "Invalid parameter or argument",
                "suggested_action": "Check input parameters for correctness"
            },
            "server_error": {
                "keywords": ["server", "500", "503", "502", "internal"],
                "severity": "MEDIUM",
                "diagnosis": "Server-side error",
                "suggested_action": "Wait and retry later"
            },
            "market_closed": {
                "keywords": ["market closed", "not trading", "suspended"],
                "severity": "MEDIUM",
                "diagnosis": "Market is currently closed or trading is suspended",
                "suggested_action": "Wait for market to open or check trading schedule"
            },
            "order_rejected": {
                "keywords": ["rejected", "cancel", "not accepted"],
                "severity": "HIGH",
                "diagnosis": "Order was rejected by the exchange",
                "suggested_action": "Check order parameters and exchange rules"
            },
            "api_changed": {
                "keywords": ["deprecated", "not found", "method not allowed", "404", "405"],
                "severity": "HIGH",
                "diagnosis": "API endpoint may have changed or been deprecated",
                "suggested_action": "Check API documentation for updates"
            },
            "data_error": {
                "keywords": ["data", "parse", "json", "format"],
                "severity": "MEDIUM",
                "diagnosis": "Error parsing or processing data",
                "suggested_action": "Check data format and processing logic"
            }
        }
        
        try:
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        # File is empty, treat as if it doesn't exist
                        raise json.JSONDecodeError("Empty file", content, 0)
                    return json.loads(content)
            else:
                # Create the file with default patterns
                with open(patterns_file, 'w', encoding='utf-8') as f:
                    json.dump(default_patterns, f, indent=4)
                return default_patterns
        except (IOError, json.JSONDecodeError, TypeError) as e:
            self.logger.log_error(f"Error loading error_patterns.json: {e}. Recreating with defaults.", "CONFIG_ERROR")
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(default_patterns, f, indent=4)
            return default_patterns
    
    def diagnose_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Provide detailed diagnosis of an error with context."""
        error_type = self.classify_error(error, context)
        error_str = str(error).lower()
        stack_trace = traceback.format_exc()
        
        # Determine severity and find matching pattern
        severity = ErrorSeverity.MEDIUM  # Default severity
        diagnosis = "Unknown error occurred"
        suggested_action = "Check logs for more details"
        
        # Match against known patterns
        matched_pattern_name = None
        matched_pattern = None
        for pattern_name, pattern in self.error_patterns.items():
            if any(keyword.lower() in error_str for keyword in pattern["keywords"]):
                matched_pattern = pattern
                severity = ErrorSeverity(pattern["severity"])
                diagnosis = pattern["diagnosis"]
                suggested_action = pattern["suggested_action"]
                matched_pattern_name = pattern_name
                break
        
        # Enhance diagnosis with context
        if context:
            if 'symbol' in context:
                diagnosis += f" for symbol {context['symbol']}"
            if 'operation' in context:
                diagnosis += f" during {context['operation']}"
        
        # Create diagnostic info
        diagnostic_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type.value,
            'severity': severity.value,
            'error_message': str(error),
            'diagnosis': diagnosis,
            'suggested_action': suggested_action,
            'context': context or {},
            'stack_trace': stack_trace,
            'matched_pattern': matched_pattern_name
        }
        
        # Store diagnostic info
        self.recent_diagnostics.append(diagnostic_info)
        
        # Log diagnostic info
        self.logger.log_error(
            f"{severity.value}: {diagnosis}", 
            error_type.value,
            {
                'diagnostic_info': diagnostic_info,
                'suggested_action': suggested_action
            }
        )
        
        # Update monitoring dashboard
        if self.monitoring_dashboard:
            try:
                self.monitoring_dashboard.log_error_alert({
                    'timestamp': datetime.now().isoformat(),
                    'error_type': error_type.value,
                    'severity': severity.value,
                    'message': diagnosis,
                    'suggested_action': suggested_action
                })
            except Exception as e:
                self.logger.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
        
        return diagnostic_info
    
    def handle_error_with_diagnostics(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error with enhanced diagnostics."""
        # Get basic error handling info
        error_info = self.handle_error(error, operation, context)
        
        # Add diagnostic information
        diagnostic_info = self.diagnose_error(error, context)
        error_info.update({
            'diagnostic_info': diagnostic_info,
            'user_message': self.get_user_friendly_message(diagnostic_info)
        })
        
        return error_info
    
    def get_user_friendly_message(self, diagnostic_info: Dict[str, Any]) -> str:
        """Generate a user-friendly error message."""
        severity = diagnostic_info['severity']
        diagnosis = diagnostic_info['diagnosis']
        action = diagnostic_info['suggested_action']
        
        if severity == ErrorSeverity.CRITICAL.value:
            prefix = "Critical Error: "
        elif severity == ErrorSeverity.HIGH.value:
            prefix = "Important: "
        elif severity == ErrorSeverity.MEDIUM.value:
            prefix = "Warning: "
        elif severity == ErrorSeverity.LOW.value:
            prefix = "Notice: "
        else:  # INFO
            prefix = "Info: "
        
        return f"{prefix}{diagnosis}. {action}."
    
    def execute_with_diagnostics(self, func: Callable, operation: str, 
                               api_name: str = None, context: Dict[str, Any] = None) -> Any:
        """Execute function with enhanced diagnostic error handling."""
        context = context or {}
        last_error = None
        start_time = time.time()
        
        # Log the operation start
        self.logger.log_system_event("OPERATION_START", {
            'operation': operation,
            'api_name': api_name,
            'context': context
        })
        
        for attempt in range(self.max_retries + 1):
            try:
                context['attempt'] = attempt
                
                if attempt > 0:
                    self.logger.log_system_event("RETRY_ATTEMPT", {
                        'operation': operation,
                        'attempt': attempt,
                        'max_retries': self.max_retries
                    })
                
                # Use circuit breaker if API name provided
                if api_name:
                    circuit_breaker = self.get_circuit_breaker(api_name)
                    result = circuit_breaker.call(func)
                else:
                    result = func()
                
                # Log successful operation
                execution_time = (time.time() - start_time) * 1000  # ms
                self.logger.log_system_event("OPERATION_SUCCESS", {
                    'operation': operation,
                    'execution_time_ms': execution_time,
                    'attempts': attempt + 1
                })
                
                # Update monitoring dashboard
                if self.monitoring_dashboard:
                    try:
                        self.monitoring_dashboard.system_metrics.operation_success_rate = 1.0
                        if 'api' in operation.lower():
                            self.monitoring_dashboard.system_metrics.api_response_time = execution_time
                    except Exception as e:
                        self.logger.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
                
                return result
                    
            except Exception as e:
                last_error = e
                context['api_name'] = api_name
                
                # Enhanced error handling with diagnostics
                error_info = self.handle_error_with_diagnostics(e, operation, context)
                
                # If this is the last attempt or shouldn't retry, break the loop
                if attempt >= self.max_retries or not error_info['should_retry']:
                    self.logger.log_system_event("OPERATION_FAILED", {
                        'operation': operation,
                        'attempts': attempt + 1,
                        'error_type': error_info['error_type'],
                        'diagnostic_info': error_info['diagnostic_info']
                    })
                    break
                
                # Wait before retrying
                retry_delay = error_info['retry_delay']
                if retry_delay > 0:
                    self.logger.log_debug(f"Waiting {retry_delay}s before retry", {
                        'operation': operation,
                        'attempt': attempt,
                        'retry_delay': retry_delay
                    })
                    time.sleep(retry_delay)
        
        # If we've exhausted all retries, update metrics and raise the last error
        if last_error:
            if self.monitoring_dashboard:
                try:
                    self.monitoring_dashboard.system_metrics.operation_success_rate = 0.0
                    self.monitoring_dashboard.system_metrics.error_count += 1
                except Exception as e:
                    self.logger.log_error(f"Failed to update monitoring dashboard: {str(e)}", "INTEGRATION_ERROR")
            
            raise last_error
    
    def get_recent_diagnostics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent error diagnostics."""
        with self.lock:
            return list(self.recent_diagnostics)[-limit:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors with diagnostics."""
        with self.lock:
            if not self.recent_diagnostics:
                return {'total_errors': 0}
            
            error_counts = {}
            severity_counts = {}
            pattern_counts = {}
            
            for diag in self.recent_diagnostics:
                error_type = diag['error_type']
                severity = diag['severity']
                pattern = diag.get('matched_pattern')
                
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                if pattern:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            return {
                'total_errors': len(self.recent_diagnostics),
                'error_counts': error_counts,
                'severity_counts': severity_counts,
                'pattern_counts': pattern_counts,
                'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None,
                'most_severe_count': severity_counts.get(ErrorSeverity.CRITICAL.value, 0) + severity_counts.get(ErrorSeverity.HIGH.value, 0),
                'circuit_breaker_states': {name: cb.state.value for name, cb in self.api_circuit_breakers.items()}
            }
    
    def add_custom_error_pattern(self, pattern_name: str, keywords: List[str], 
                               severity: str, diagnosis: str, suggested_action: str) -> bool:
        """Add a custom error pattern for future diagnostics."""
        try:
            # Validate severity
            ErrorSeverity(severity)  # Will raise ValueError if invalid
            
            # Add pattern
            self.error_patterns[pattern_name] = {
                "keywords": keywords,
                "severity": severity,
                "diagnosis": diagnosis,
                "suggested_action": suggested_action
            }
            
            # Save to file
            patterns_file = os.path.join(os.path.dirname(__file__), 'error_patterns.json')
            with open(patterns_file, 'w') as f:
                json.dump(self.error_patterns, f, indent=4)
            
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to add custom error pattern: {str(e)}", "CONFIG_ERROR")
            return False