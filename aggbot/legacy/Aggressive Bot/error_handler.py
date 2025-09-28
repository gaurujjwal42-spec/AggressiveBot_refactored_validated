# -*- coding: utf-8 -*-
"""
Advanced Error Handling System for Trading Bot

Provides comprehensive error handling for:
- Exchange API compliance
- Market connectivity issues
- Network timeouts and retries
- Rate limiting management
- Circuit breaker patterns
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import deque
from enum import Enum
import requests

class ErrorType(Enum):
    """Types of errors that can occur."""
    NETWORK_ERROR = "NETWORK_ERROR"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    EXCHANGE_ERROR = "EXCHANGE_ERROR"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    INVALID_PAIR = "INVALID_PAIR"
    SLIPPAGE_EXCEEDED = "SLIPPAGE_EXCEEDED"
    GAS_ESTIMATION_FAILED = "GAS_ESTIMATION_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout))
    
    def _on_success(self):
        """Handle successful operation."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

class AdvancedErrorHandler:
    """Advanced error handling system for trading operations."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history = deque(maxlen=1000)
        self.api_circuit_breakers = {}
        self.rate_limit_trackers = {}
        self.lock = threading.Lock()
    
    def get_circuit_breaker(self, api_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for API."""
        if api_name not in self.api_circuit_breakers:
            self.api_circuit_breakers[api_name] = CircuitBreaker()
        return self.api_circuit_breakers[api_name]
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorType:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        
        if isinstance(error, requests.exceptions.ConnectionError):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, requests.exceptions.Timeout):
            return ErrorType.NETWORK_ERROR
        elif "429" in error_str or "rate limit" in error_str:
            return ErrorType.API_RATE_LIMIT
        elif "insufficient" in error_str and "funds" in error_str:
            return ErrorType.INSUFFICIENT_FUNDS
        elif "invalid pair" in error_str or "symbol not found" in error_str:
            return ErrorType.INVALID_PAIR
        elif "slippage" in error_str:
            return ErrorType.SLIPPAGE_EXCEEDED
        elif "gas" in error_str and "estimation" in error_str:
            return ErrorType.GAS_ESTIMATION_FAILED
        elif "transaction failed" in error_str or "reverted" in error_str:
            return ErrorType.TRANSACTION_FAILED
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error with appropriate strategy."""
        error_type = self.classify_error(error, context)
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_type': error_type.value,
            'error_message': str(error),
            'context': context or {},
            'should_retry': False,
            'retry_delay': 0,
            'action_taken': 'NONE'
        }
        
        # Determine handling strategy based on error type
        if error_type == ErrorType.NETWORK_ERROR:
            error_info['should_retry'] = True
            error_info['retry_delay'] = self._calculate_backoff_delay(context.get('attempt', 0))
            error_info['action_taken'] = 'RETRY_WITH_BACKOFF'
        
        elif error_type == ErrorType.API_RATE_LIMIT:
            api_name = context.get('api_name', 'unknown')
            delay = self._handle_rate_limit(api_name, error)
            error_info['should_retry'] = True
            error_info['retry_delay'] = delay
            error_info['action_taken'] = 'RATE_LIMIT_BACKOFF'
        
        elif error_type == ErrorType.INSUFFICIENT_FUNDS:
            error_info['should_retry'] = False
            error_info['action_taken'] = 'SKIP_TRADE'
        
        elif error_type == ErrorType.INVALID_PAIR:
            error_info['should_retry'] = False
            error_info['action_taken'] = 'BLACKLIST_PAIR'
        
        elif error_type == ErrorType.SLIPPAGE_EXCEEDED:
            error_info['should_retry'] = True
            error_info['retry_delay'] = 5  # Wait 5 seconds for market to stabilize
            error_info['action_taken'] = 'RETRY_WITH_ADJUSTED_SLIPPAGE'
        
        elif error_type == ErrorType.GAS_ESTIMATION_FAILED:
            error_info['should_retry'] = True
            error_info['retry_delay'] = 10  # Wait for network congestion to clear
            error_info['action_taken'] = 'RETRY_WITH_HIGHER_GAS'
        
        elif error_type == ErrorType.TRANSACTION_FAILED:
            error_info['should_retry'] = False
            error_info['action_taken'] = 'LOG_AND_CONTINUE'
        
        else:  # UNKNOWN_ERROR
            error_info['should_retry'] = True
            attempt = context.get('attempt', 0) if context else 0
            error_info['retry_delay'] = self._calculate_backoff_delay(attempt)
            error_info['action_taken'] = 'CAUTIOUS_RETRY'
        
        # Log error
        with self.lock:
            self.error_history.append(error_info)
        
        return error_info
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return min(self.base_delay * (2 ** attempt), 60)  # Cap at 60 seconds
    
    def _handle_rate_limit(self, api_name: str, error: Exception) -> float:
        """Handle rate limiting with appropriate delay."""
        # Extract retry-after header if available
        retry_after = 60  # Default 1 minute
        
        if hasattr(error, 'response') and error.response:
            retry_after = int(error.response.headers.get('Retry-After', retry_after))
        
        # Track rate limit for this API
        if api_name not in self.rate_limit_trackers:
            self.rate_limit_trackers[api_name] = {
                'last_rate_limit': datetime.now(),
                'rate_limit_count': 0
            }
        
        tracker = self.rate_limit_trackers[api_name]
        tracker['last_rate_limit'] = datetime.now()
        tracker['rate_limit_count'] += 1
        
        # Increase delay if we're hitting rate limits frequently
        if tracker['rate_limit_count'] > 3:
            retry_after *= 2
        
        return retry_after
    
    def execute_with_retry(self, func: Callable, operation: str, 
                          api_name: str = None, context: Dict[str, Any] = None) -> Any:
        """Execute function with retry logic and error handling."""
        context = context or {}
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                context['attempt'] = attempt

                # Use circuit breaker if API name provided
                if api_name:
                    circuit_breaker = self.get_circuit_breaker(api_name)
                    return circuit_breaker.call(func)
                else:
                    return func()

            except Exception as e:
                last_error = e
                context['api_name'] = api_name
                error_info = self.handle_error(e, operation, context)

                # If this is the last attempt or shouldn't retry, break the loop
                if attempt >= self.max_retries or not error_info['should_retry']:
                    break

                # Wait before retrying
                retry_delay = error_info['retry_delay']
                if retry_delay > 0:
                    time.sleep(retry_delay)

        # If we've exhausted all retries, raise the last error
        if last_error:
            raise last_error
            
        # This should never be reached, but just in case
        raise Exception(f"Max retries exceeded for operation: {operation}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        with self.lock:
            if not self.error_history:
                return {'total_errors': 0}
            
            error_counts = {}
            recent_errors = []
            now = datetime.now()
            
            for error in self.error_history:
                error_type = error['error_type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                # Count errors in last hour
                error_time = datetime.fromisoformat(error['timestamp'])
                if now - error_time < timedelta(hours=1):
                    recent_errors.append(error)
            
            return {
                'total_errors': len(self.error_history),
                'error_counts': error_counts,
                'recent_errors_count': len(recent_errors),
                'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None,
                'circuit_breaker_states': {name: cb.state.value for name, cb in self.api_circuit_breakers.items()}
            }
    
    def reset_circuit_breaker(self, api_name: str):
        """Manually reset circuit breaker for an API."""
        if api_name in self.api_circuit_breakers:
            cb = self.api_circuit_breakers[api_name]
            with cb.lock:
                cb.failure_count = 0
                cb.state = CircuitBreakerState.CLOSED
                cb.last_failure_time = None
    
    def is_api_healthy(self, api_name: str) -> bool:
        """Check if API is healthy (circuit breaker not open)."""
        if api_name not in self.api_circuit_breakers:
            return True
        return self.api_circuit_breakers[api_name].state != CircuitBreakerState.OPEN