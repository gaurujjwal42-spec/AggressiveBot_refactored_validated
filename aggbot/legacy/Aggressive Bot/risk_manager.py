# -*- coding: utf-8 -*-
"""Advanced Risk Management Module for Trading Bot

This module provides comprehensive risk analysis and management protocols
to ensure safe trading operations and optimal risk-adjusted returns.
"""

import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json

from trading_utils import calculate_volatility

class RiskManager:
    """Comprehensive risk management system for trading operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trade_history = deque(maxlen=1000)  # Store last 1000 trades
        self.daily_pnl = deque(maxlen=30)  # Store last 30 days P&L
        self.position_sizes = deque(maxlen=100)  # Track position sizing
        self.drawdown_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)
        
        # Risk thresholds
        self.max_daily_loss_pct = config.get('MAX_DAILY_LOSS_PCT', 5.0)
        self.max_portfolio_risk_pct = config.get('MAX_PORTFOLIO_RISK_PCT', 15.0)
        self.max_single_position_pct = config.get('MAX_SINGLE_POSITION_PCT', 10.0)
        self.max_drawdown_pct = config.get('MAX_DRAWDOWN_PCT', 20.0)
        self.min_sharpe_ratio = config.get('MIN_SHARPE_RATIO', 0.5)
        
        # Volatility multiplier thresholds
        self.volatility_multiplier_high_threshold = config.get('VOLATILITY_MULTIPLIER_HIGH_THRESHOLD', 1.5)
        self.volatility_multiplier_medium_threshold = config.get('VOLATILITY_MULTIPLIER_MEDIUM_THRESHOLD', 1.2)
        self.concentration_similarity_threshold = config.get('CONCENTRATION_SIMILARITY_THRESHOLD', 0.6)

        # Position sizing parameters
        self.kelly_fraction = config.get('KELLY_FRACTION', 0.25)
        self.volatility_lookback = config.get('VOLATILITY_LOOKBACK_DAYS', 14)
        
        # Risk state tracking
        self.daily_start_balance = 0
        self.current_portfolio_value = 0
        self.peak_balance = 0
        self.current_drawdown = 0
        self.risk_level = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
        self.emergency_position_size_reduction = False  # Emergency risk reduction flag
        
        # Enhanced live trading safety features
        self.circuit_breaker_triggered = False
        self.circuit_breaker_cooldown = 0
        self.circuit_breaker_volatility_cooldown = config.get('CIRCUIT_BREAKER_VOLATILITY_COOLDOWN', 300) # 5 min
        self.circuit_breaker_losses_cooldown = config.get('CIRCUIT_BREAKER_LOSSES_COOLDOWN', 600) # 10 min
        self.circuit_breaker_drawdown_cooldown = config.get('CIRCUIT_BREAKER_DRAWDOWN_COOLDOWN', 1800) # 30 min
        self.max_consecutive_losses = config.get('MAX_CONSECUTIVE_LOSSES', 5)
        self.volatility_circuit_breaker = config.get('VOLATILITY_CIRCUIT_BREAKER', 0.15)
        self.correlation_risk_threshold = config.get('CORRELATION_RISK_THRESHOLD', 0.8) # Not used, but good to have
        self.position_concentration_limit = config.get('POSITION_CONCENTRATION_LIMIT', 0.3)
        self.emergency_exit_threshold = config.get('EMERGENCY_EXIT_THRESHOLD', 0.25)
        
        # Advanced monitoring
        self.risk_alerts = deque(maxlen=50)
        self.performance_degradation_counter = 0
        self.last_risk_assessment = datetime.now()
        
        logging.info("Enhanced Risk Manager initialized with live trading safety protocols")

    def start_new_day(self, opening_balance: float):
        """Resets daily risk metrics at the start of a new trading day."""
        self.daily_start_balance = opening_balance
        logging.info(f"Risk Manager: New trading day started with balance: ${opening_balance:.2f}")
    
    def _calculate_base_risk_usd(self, symbol: str, portfolio_value: float, confidence_score: float, prices: List[float]) -> float:
        """Calculates the base USD amount to risk on a trade."""
        base_risk_pct = self.config.get('BASE_RISK_PER_TRADE_PCT', 2.0) / 100
        volatility_multiplier = self._get_volatility_multiplier(symbol, prices)
        kelly_multiplier = self._calculate_kelly_multiplier(confidence_score)
        
        risk_per_trade_usd = portfolio_value * base_risk_pct * volatility_multiplier * kelly_multiplier
        return risk_per_trade_usd

    def _calculate_size_from_risk(
        self, risk_per_trade_usd: float, entry_price: float, stop_loss_price: float, portfolio_value: float
    ) -> float:
        """Calculates position size based on stop-loss distance."""
        if entry_price > 0 and stop_loss_price > 0 and entry_price > stop_loss_price:
            risk_per_unit_pct = (entry_price - stop_loss_price) / entry_price
            if risk_per_unit_pct > 0:
                return risk_per_trade_usd / risk_per_unit_pct
        
        # Fallback to a simple percentage of portfolio if stop-loss is invalid
        logging.warning(f"Invalid stop-loss or entry price ({entry_price=}); falling back to default position sizing.")
        base_risk_pct = self.config.get('BASE_RISK_PER_TRADE_PCT', 2.0) / 100
        return portfolio_value * (base_risk_pct * 2)

    def _apply_position_constraints(
        self, position_size_usd: float, portfolio_value: float, symbol: str
    ) -> float:
        """Applies emergency reductions and min/max position size limits."""
        # Apply emergency position size reduction if critical risks detected
        if getattr(self, 'emergency_position_size_reduction', False):
            position_size_usd *= 0.5
            logging.warning(
                f"Emergency position size reduction applied for {symbol}: "
                "50% reduction due to critical risk conditions"
            )

        # Apply maximum and minimum position size limits
        max_position_usd = portfolio_value * (self.max_single_position_pct / 100)
        min_position_usd = self.config.get('MIN_TRADE_SIZE_USDT', 5)

        constrained_size = max(min_position_usd, min(position_size_usd, max_position_usd))
        return constrained_size

    def _record_position_size(self, symbol: str, size_usd: float, portfolio_value: float):
        """Records the calculated position size for analysis."""
        if portfolio_value > 0:
            portfolio_pct = (size_usd / portfolio_value) * 100
        else:
            portfolio_pct = 0

        self.position_sizes.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'size_usd': size_usd,
            'portfolio_pct': portfolio_pct
        })

    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              portfolio_value: float,
                              confidence_score: float = 0.5,
                              market_data: Dict[str, Any] = None) -> float:
        """
        Calculate optimal position size using multiple risk models by calling
        helper methods for each step of the calculation.
        """
        try:
            prices = market_data.get('prices', []) if market_data else []

            # 1. Determine the base USD amount to risk on the trade
            risk_per_trade_usd = self._calculate_base_risk_usd(
                symbol, portfolio_value, confidence_score, prices
            )

            # 2. Calculate the initial position size based on the risk and stop-loss
            position_size_usd = self._calculate_size_from_risk(
                risk_per_trade_usd, entry_price, stop_loss_price, portfolio_value
            )

            # 3. Apply constraints (emergency reduction, min/max limits)
            final_position_size_usd = self._apply_position_constraints(
                position_size_usd, portfolio_value, symbol
            )

            # 4. Record and log the final position size
            self._record_position_size(
                symbol, final_position_size_usd, portfolio_value
            )
            
            portfolio_pct = (final_position_size_usd / portfolio_value) * 100 if portfolio_value > 0 else 0
            logging.info(
                f"Position size calculated for {symbol}: ${final_position_size_usd:.2f} "
                f"({portfolio_pct:.2f}% of portfolio)"
            )
            return final_position_size_usd
            
        except Exception as e:
            logging.error(f"Error calculating position size for {symbol}: {e}")
            return self.config.get('MIN_TRADE_SIZE_USDT', 5)
    
    def _get_volatility_multiplier(self, symbol: str, prices: List[float]) -> float:
        """Calculate volatility-based position size multiplier"""
        try:
            recent_volatility = calculate_volatility(prices)
            self.volatility_history.append(recent_volatility)

            if len(self.volatility_history) < 20:
                return 1.0 # Not enough data, use default

            # Adaptive volatility based on moving average
            avg_volatility = statistics.mean(self.volatility_history)
            
            if recent_volatility > avg_volatility * self.volatility_multiplier_high_threshold:
                return 0.5
            elif recent_volatility > avg_volatility * self.volatility_multiplier_medium_threshold:
                return 0.75
            else:  # Normal or low volatility
                return 1.0
                
        except Exception as e:
            logging.warning(f"Could not calculate volatility multiplier for {symbol}: {e}")
            return 1.0  # Default multiplier
    
    def _calculate_kelly_multiplier(self, confidence_score: float) -> float:
        """Calculate Kelly Criterion multiplier based on confidence"""
        try:
            # Simplified Kelly: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            
            win_rate = self._get_recent_win_rate()
            avg_win = self._get_average_win()
            avg_loss = self._get_average_loss()
            
            if avg_loss == 0 or win_rate == 0:
                return 0.5  # Conservative default
            
            # Calculate Kelly fraction
            odds_ratio = abs(avg_win / avg_loss)
            # Prevent division by zero if odds_ratio is 0
            if odds_ratio == 0:
                return 0.5 # Avoid division by zero, return conservative default

            kelly_fraction = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
            
            # Apply confidence adjustment and cap at reasonable levels
            kelly_adjusted = kelly_fraction * confidence_score * self.kelly_fraction
            return max(0.1, min(kelly_adjusted, 1.0))
            
        except Exception:
            return 0.5  # Conservative default
    
    def assess_trade_risk(self, 
                         symbol: str, 
                         trade_type: str, 
                         position_size_usd: float,
                         portfolio_value: float,
                         open_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trade risk assessment"""
        
        risk_assessment = {
            'approved': True,
            'risk_level': 'LOW',
            'risk_factors': [],
            'recommendations': [],
            'max_position_size': position_size_usd
        }
        
        try:
            # 1. Portfolio concentration risk
            position_pct = (position_size_usd / portfolio_value) * 100
            if position_pct > self.max_single_position_pct:
                risk_assessment['risk_factors'].append(f"Position size ({position_pct:.1f}%) exceeds maximum ({self.max_single_position_pct}%)")
                risk_assessment['max_position_size'] = portfolio_value * (self.max_single_position_pct / 100)
                risk_assessment['risk_level'] = 'HIGH'
            
            # 2. Daily loss limit check
            daily_pnl_pct = self._calculate_daily_pnl_pct(portfolio_value)
            if daily_pnl_pct <= -self.max_daily_loss_pct:
                risk_assessment['approved'] = False
                risk_assessment['risk_factors'].append(f"Daily loss limit reached ({daily_pnl_pct:.1f}%)")
                risk_assessment['risk_level'] = 'CRITICAL'
            
            # 3. Maximum drawdown check
            if self.current_drawdown > self.max_drawdown_pct:
                risk_assessment['approved'] = False
                risk_assessment['risk_factors'].append(f"Maximum drawdown exceeded ({self.current_drawdown:.1f}%)")
                risk_assessment['risk_level'] = 'CRITICAL'
            
            # 4. Portfolio heat check (total risk exposure)
            total_risk_exposure = self._calculate_total_risk_exposure(portfolio_value, open_positions)
            if total_risk_exposure > self.max_portfolio_risk_pct:
                risk_assessment['risk_factors'].append(f"Total portfolio risk ({total_risk_exposure:.1f}%) too high")
                risk_assessment['risk_level'] = 'HIGH'
            
            # 5. Recent performance check
            recent_sharpe = self._calculate_recent_sharpe_ratio()
            if recent_sharpe < self.min_sharpe_ratio:
                risk_assessment['risk_factors'].append(f"Poor recent performance (Sharpe: {recent_sharpe:.2f})")
                if risk_assessment['risk_level'] == 'LOW':
                    risk_assessment['risk_level'] = 'MEDIUM'
            
            # 6. Generate recommendations
            if risk_assessment['risk_factors']:
                risk_assessment['recommendations'] = self._generate_risk_recommendations(risk_assessment)
            
            logging.info(f"Risk assessment for {symbol}: {risk_assessment['risk_level']} risk, Approved: {risk_assessment['approved']}")
            return risk_assessment
            
        except Exception as e:
            logging.error(f"Error in risk assessment: {e}")
            return {
                'approved': False,
                'risk_level': 'CRITICAL',
                'risk_factors': ['Risk assessment system error'],
                'recommendations': ['Manual review required'],
                'max_position_size': 0
            }
    
    def update_trade_result(self, trade_result: Dict[str, Any]):
        """Update risk metrics with new trade result"""
        try:
            self.trade_history.append({
                'timestamp': trade_result.get('timestamp', datetime.now()),
                'symbol': trade_result.get('symbol', ''),
                'type': trade_result.get('type', ''),
                'pnl': trade_result.get('pnl', 0),
                'pnl_pct': trade_result.get('pnl_pct', 0)
            })
            
            # Update daily P&L tracking
            self._update_daily_pnl(trade_result.get('pnl', 0))
            
            logging.info(f"Risk metrics updated with trade result: {trade_result.get('symbol', '')} P&L: ${trade_result.get('pnl', 0):.2f}")
            
        except Exception as e:
            logging.error(f"Error updating trade result in risk manager: {e}")

    # Helper methods
    def _get_recent_win_rate(self) -> float:
        """Calculate recent win rate from trade history"""
        if len(self.trade_history) < 20:
            return 0.5  # Default assumption
        
        recent_trades = list(self.trade_history) # Use all available trades for win rate
        winning_trades = sum(1 for trade in recent_trades if trade['pnl'] > 0)

        return winning_trades / len(recent_trades)
    
    def _get_average_win(self) -> float:
        """Calculate average winning trade amount from recent trades."""
        recent_trades = list(self.trade_history)[-30:]
        winning_trades = [trade['pnl'] for trade in recent_trades if trade['pnl'] > 0]
        return statistics.mean(winning_trades) if winning_trades else 0
    
    def _get_average_loss(self) -> float:
        """Calculate average losing trade amount from recent trades."""
        recent_trades = list(self.trade_history)[-30:]
        losing_trades = [abs(trade['pnl']) for trade in recent_trades if trade['pnl'] < 0]
        return statistics.mean(losing_trades) if losing_trades else 0
    
    def _calculate_daily_pnl_pct(self, portfolio_value: float) -> float:
        """Calculate today's P&L percentage"""
        if self.daily_start_balance == 0:
            return 0
        return ((portfolio_value - self.daily_start_balance) / self.daily_start_balance) * 100
    
    def _calculate_total_risk_exposure(self, portfolio_value: float, open_positions: Dict[str, Any]) -> float:
        """
        Calculate total portfolio risk exposure by summing the potential loss
        of each open position if its stop-loss is hit.
        """
        if not open_positions or portfolio_value == 0:
            return 0.0

        total_risk_usd = 0.0
        # FIX: Use abs() to ensure the percentage is positive, making the logic
        # robust even if the config value is accidentally positive.
        fixed_stop_loss_pct = abs(self.config.get('STOP_LOSS_PCT', -10.0)) / 100.0

        for pos in open_positions.values():
            entry_price = pos.get('entry_price')
            stop_loss_price = pos.get('stop_loss_price')
            amount = pos.get('token_amount')
            usdt_invested = pos.get('usdt_invested')

            if not all([entry_price, amount, usdt_invested]):
                continue  # Skip positions with incomplete data

            # Prioritize ATR-based stop-loss if it's valid
            if stop_loss_price and stop_loss_price > 0:
                position_risk_usd = (entry_price - stop_loss_price) * amount
                total_risk_usd += position_risk_usd
            else:
                # Fallback to fixed percentage stop-loss
                position_risk_usd = usdt_invested * fixed_stop_loss_pct
                total_risk_usd += position_risk_usd
        
        total_risk_pct = (total_risk_usd / portfolio_value) * 100
        return min(total_risk_pct, 100.0)
    
    def _update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        today = datetime.now().date()
        if not self.daily_pnl or self.daily_pnl[-1]['date'] != today:
            self.daily_pnl.append({'date': today, 'pnl': pnl})
        else:
            self.daily_pnl[-1]['pnl'] += pnl
    
    def _get_avg_position_size_pct(self) -> float:
        """Get average position size as percentage of portfolio"""
        if not self.position_sizes:
            return 0
        return statistics.mean([pos['portfolio_pct'] for pos in self.position_sizes])
    
    def _calculate_risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return metric"""
        if len(self.trade_history) < 5:
            return 0
        
        total_return = sum(trade['pnl'] for trade in self.trade_history)
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 1
        
        return total_return / max_drawdown if max_drawdown > 0 else 0
    
    def _generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_assessment['risk_level'] == 'HIGH':
            recommendations.extend([
                "Reduce position sizes by 50%",
                "Implement tighter stop losses",
                "Consider taking profits on existing positions"
            ])
        elif risk_assessment['risk_level'] == 'CRITICAL':
            recommendations.extend([
                "Stop all new trading immediately",
                "Close all losing positions",
                "Review and adjust risk parameters",
                "Wait for market conditions to improve"
            ])
        
        return recommendations
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted due to risk conditions"""
        try:
            # Check daily loss limit against current portfolio value
            if self._calculate_daily_pnl_pct(self.current_portfolio_value) <= -self.max_daily_loss_pct:
                return True, f"Daily loss limit exceeded ({self.max_daily_loss_pct}%)"
            
            # Check maximum drawdown
            if self.current_drawdown > self.max_drawdown_pct:
                return True, f"Maximum drawdown exceeded ({self.max_drawdown_pct}%)"
            
            # Check consecutive losses
            if len(self.trade_history) >= self.max_consecutive_losses:
                last_n_trades = list(self.trade_history)[-self.max_consecutive_losses:]
                if all(trade['pnl'] < 0 for trade in last_n_trades):
                    return True, f"{self.max_consecutive_losses} consecutive losing trades detected"
            
            return False, "All risk checks passed"
            
        except Exception as e:
            logging.error(f"Error in trading halt check: {e}")
            return True, "Risk system error - halting for safety"
    
    def update_portfolio_value(self, portfolio_value: float):
        """Update current portfolio value for risk calculations"""
        self.current_portfolio_value = portfolio_value
        
        # Update peak balance tracking
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - portfolio_value) / self.peak_balance) * 100
    
    def get_risk_metrics(self, open_positions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive risk metrics for real-time monitoring"""
        if open_positions is None:
            open_positions = {}
        try:
            # Calculate daily P&L percentage
            daily_pnl_pct = self._calculate_daily_pnl_pct(getattr(self, 'current_portfolio_value', 0))
            
            # Calculate recent Sharpe ratio
            recent_sharpe = self._calculate_recent_sharpe_ratio()
            
            # Calculate total portfolio risk exposure
            total_risk_exposure_pct = self._calculate_total_risk_exposure(self.current_portfolio_value, open_positions)
            
            # Determine current risk level
            risk_level = self._determine_risk_level(daily_pnl_pct, self.current_drawdown, recent_sharpe)
            
            return {
                'daily_pnl_pct': daily_pnl_pct,
                'current_drawdown_pct': self.current_drawdown,
                'total_risk_exposure_pct': total_risk_exposure_pct,
                'recent_sharpe_ratio': recent_sharpe,
                'risk_level': risk_level,
                'peak_balance': self.peak_balance,
                'current_portfolio_value': getattr(self, 'current_portfolio_value', 0)
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {
                'daily_pnl_pct': 0,
                'current_drawdown_pct': 0,
                'total_risk_exposure_pct': 0,
                'recent_sharpe_ratio': 0,
                'risk_level': 'UNKNOWN'
            }
    

    
    def _calculate_recent_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for recent trades based on percentage returns."""
        if len(self.trade_history) < 10:
            return 0.0

        # Use last 30 trades for a more stable metric
        recent_trades = list(self.trade_history)[-30:]
        returns = [
            trade['pnl_pct'] for trade in recent_trades if 'pnl_pct' in trade and trade['pnl_pct'] is not None
        ]

        if len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        # Simplified Sharpe ratio (assuming risk-free rate = 0)
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _determine_risk_level(self, daily_pnl_pct: float, drawdown_pct: float, sharpe_ratio: float) -> str:
        """Determine current risk level based on multiple factors"""
        risk_score = 0
        
        # Daily P&L factor
        if daily_pnl_pct <= -self.max_daily_loss_pct * 0.8:
            risk_score += 3
        elif daily_pnl_pct <= -self.max_daily_loss_pct * 0.5:
            risk_score += 2
        elif daily_pnl_pct <= -self.max_daily_loss_pct * 0.3:
            risk_score += 1
        
        # Drawdown factor
        if drawdown_pct >= self.max_drawdown_pct * 0.8:
            risk_score += 3
        elif drawdown_pct >= self.max_drawdown_pct * 0.5:
            risk_score += 2
        elif drawdown_pct >= self.max_drawdown_pct * 0.3:
            risk_score += 1
        
        # Sharpe ratio factor
        if sharpe_ratio < self.min_sharpe_ratio * 0.5:
            risk_score += 2
        elif sharpe_ratio < self.min_sharpe_ratio:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return 'CRITICAL'
        elif risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def check_circuit_breaker(self, market_volatility: float = None) -> Tuple[bool, str]:
        """Enhanced circuit breaker system for live trading safety"""
        try:
            current_time = time.time()
            
            # Check if circuit breaker is in cooldown
            if self.circuit_breaker_triggered and current_time < self.circuit_breaker_cooldown:
                remaining_cooldown = int(self.circuit_breaker_cooldown - current_time)
                return True, f"Circuit breaker active - cooldown: {remaining_cooldown}s"
            
            # Reset circuit breaker if cooldown expired
            if self.circuit_breaker_triggered and current_time >= self.circuit_breaker_cooldown:
                self.circuit_breaker_triggered = False
                logging.info("Circuit breaker cooldown expired - trading resumed")
            
            # Check volatility circuit breaker
            if market_volatility and market_volatility > self.volatility_circuit_breaker:
                self._trigger_circuit_breaker("High market volatility detected", self.circuit_breaker_volatility_cooldown)
                return True, f"Volatility circuit breaker triggered: {market_volatility:.3f}"
            
            # Check consecutive losses
            if len(self.trade_history) >= self.max_consecutive_losses:
                recent_trades = list(self.trade_history)[-self.max_consecutive_losses:]
                if all(trade['pnl'] < 0 for trade in recent_trades):
                    self._trigger_circuit_breaker("Consecutive losses detected", self.circuit_breaker_losses_cooldown)
                    return True, f"Consecutive loss circuit breaker triggered"
            
            # Check emergency drawdown threshold
            if self.current_drawdown > self.emergency_exit_threshold * 100:
                self._trigger_circuit_breaker("Emergency drawdown threshold exceeded", self.circuit_breaker_drawdown_cooldown)
                return True, f"Emergency exit triggered - drawdown: {self.current_drawdown:.2f}%"
            
            return False, "Circuit breaker checks passed"
            
        except Exception as e:
            logging.error(f"Error in circuit breaker check: {e}")
            return True, "Circuit breaker error - halting for safety"
    
    def _trigger_circuit_breaker(self, reason: str, cooldown_seconds: int):
        """Trigger circuit breaker with specified cooldown"""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_cooldown = time.time() + cooldown_seconds
        
        alert = {
            'timestamp': datetime.now(),
            'type': 'CIRCUIT_BREAKER',
            'reason': reason,
            'cooldown_seconds': cooldown_seconds
        }
        self.risk_alerts.append(alert)
        
        logging.warning(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason} - Cooldown: {cooldown_seconds}s")
    
    def assess_position_concentration(self, open_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """Assess if positions are too concentrated in similar assets"""
        try:
            if not open_positions or len(open_positions) <= 1:
                return False, "No concentration risk"
            
            total_value = sum(pos.get('usdt_invested', 0) for pos in open_positions.values())
            if total_value == 0:
                return False, "No position value to assess"
            
            # Check individual position concentration
            for symbol, position in open_positions.items():
                position_pct = position.get('usdt_invested', 0) / total_value
                if position_pct > self.position_concentration_limit:
                    return True, f"Position {symbol} exceeds concentration limit: {position_pct:.2%}"
            
            # Check sector/category concentration (simplified)
            # This could be enhanced with actual sector classification
            symbol_prefixes = {}
            for symbol in open_positions.keys():
                prefix = symbol[:3] if len(symbol) > 3 else symbol
                symbol_prefixes[prefix] = symbol_prefixes.get(prefix, 0) + 1
            
            max_similar = max(symbol_prefixes.values())
            if max_similar > len(open_positions) * self.concentration_similarity_threshold:
                return True, f"High concentration in similar assets: {max_similar}/{len(open_positions)}"
            
            return False, "Position concentration within limits"
            
        except Exception as e:
            logging.error(f"Error assessing position concentration: {e}")
            return True, "Concentration assessment error"
    
    def get_enhanced_risk_assessment(self, open_positions: Dict[str, Any] = None, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive risk assessment for live trading"""
        try:
            if open_positions is None:
                open_positions = {}
            
            # Get basic risk metrics
            basic_metrics = self.get_risk_metrics(open_positions)
            
            # Enhanced assessments
            market_volatility = market_data.get('volatility', 0) if market_data else 0
            circuit_breaker_active, cb_reason = self.check_circuit_breaker(market_volatility)
            concentration_risk, conc_reason = self.assess_position_concentration(open_positions)
            trading_halt, halt_reason = self.should_halt_trading()
            
            # Performance degradation check
            recent_performance = self._assess_recent_performance()
            
            # Risk score calculation
            risk_score = self._calculate_comprehensive_risk_score(
                basic_metrics, circuit_breaker_active, concentration_risk, recent_performance
            )
            
            assessment = {
                **basic_metrics,
                'circuit_breaker_active': circuit_breaker_active,
                'circuit_breaker_reason': cb_reason,
                'concentration_risk': concentration_risk,
                'concentration_reason': conc_reason,
                'trading_halt_required': trading_halt,
                'halt_reason': halt_reason,
                'market_volatility': market_volatility,
                'recent_performance_score': recent_performance,
                'comprehensive_risk_score': risk_score,
                'recommendations': self._generate_enhanced_recommendations(risk_score),
                'last_assessment': datetime.now().isoformat()
            }
            
            self.last_risk_assessment = datetime.now()
            return assessment
            
        except Exception as e:
            logging.error(f"Error in enhanced risk assessment: {e}")
            return {'error': str(e), 'risk_level': 'CRITICAL'}
    
    def _assess_recent_performance(self) -> float:
        """Assess recent trading performance trend"""
        try:
            if len(self.trade_history) < 10:
                return 0.5  # Neutral score
            
            recent_trades = list(self.trade_history)[-20:]
            recent_pnl = [trade['pnl'] for trade in recent_trades]
            
            # Calculate performance trend
            positive_trades = sum(1 for pnl in recent_pnl if pnl > 0)
            performance_ratio = positive_trades / len(recent_pnl)
            
            # Calculate average return trend
            avg_return = sum(recent_pnl) / len(recent_pnl)
            
            # Combine metrics for performance score (0-1)
            performance_score = (performance_ratio + (0.5 + avg_return / 100)) / 2
            return max(0, min(1, performance_score))
            
        except Exception:
            return 0.5
    
    def _calculate_comprehensive_risk_score(self, basic_metrics: Dict, circuit_breaker: bool, concentration_risk: bool, performance: float) -> float:
        """Calculate comprehensive risk score (0-100)"""
        try:
            score = 0
            
            # Basic risk factors (0-40 points)
            if basic_metrics['risk_level'] == 'CRITICAL':
                score += 40
            elif basic_metrics['risk_level'] == 'HIGH':
                score += 30
            elif basic_metrics['risk_level'] == 'MEDIUM':
                score += 20
            else:
                score += 10
            
            # Circuit breaker (0-20 points)
            if circuit_breaker:
                score += 20
            
            # Concentration risk (0-15 points)
            if concentration_risk:
                score += 15
            
            # Performance degradation (0-25 points)
            performance_risk = (1 - performance) * 25
            score += performance_risk
            
            return min(100, score)
            
        except Exception:
            return 75  # High risk default
    
    def _generate_enhanced_recommendations(self, risk_score: float) -> List[str]:
        """Generate enhanced risk management recommendations"""
        recommendations = []
        
        if risk_score >= 80:
            recommendations.extend([
                "🚨 CRITICAL: Stop all trading immediately",
                "Close all positions at market price",
                "Review risk parameters urgently",
                "Wait for market stabilization"
            ])
        elif risk_score >= 60:
            recommendations.extend([
                "⚠️ HIGH RISK: Reduce position sizes by 70%",
                "Tighten stop losses to 3%",
                "Close weakest performing positions",
                "Monitor market conditions closely"
            ])
        elif risk_score >= 40:
            recommendations.extend([
                "⚡ MEDIUM RISK: Reduce position sizes by 40%",
                "Implement trailing stops",
                "Avoid new high-risk trades",
                "Review portfolio diversification"
            ])
        else:
            recommendations.extend([
                "✅ LOW RISK: Normal trading operations",
                "Monitor for emerging risks",
                "Consider increasing position sizes gradually"
            ])
        
        return recommendations