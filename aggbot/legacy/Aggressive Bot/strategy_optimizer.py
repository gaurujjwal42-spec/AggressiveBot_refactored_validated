#!/usr/bin/env python3
"""
Advanced Trading Strategy Optimizer
Dynamically optimizes trading parameters based on market conditions and performance metrics
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import copy
import threading
import time

from trading_utils import calculate_volatility

@dataclass
class MarketCondition:
    """Market condition analysis"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_profile: str = "NORMAL"  # LOW, NORMAL, HIGH
    market_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    liquidity_score: float = 0.0
    correlation_btc: float = 0.0
    rsi_market: float = 50.0
    fear_greed_index: float = 50.0
    regime: str = "RANGING"  # Can be 'TRENDING' or 'RANGING'
    timestamp: str = ""

@dataclass
class OptimizedParameters:
    """Optimized trading parameters"""
    # Entry parameters
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_threshold: float = 0.0
    volume_threshold: float = 1.5
    momentum_threshold: float = 0.02
    
    # Risk management
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    max_positions: int = 5
    
    # Market condition adjustments
    volatility_adjustment: float = 1.0
    trend_following_strength: float = 0.5
    mean_reversion_strength: float = 0.5
    
    # Timing parameters
    entry_delay_seconds: int = 0
    exit_delay_seconds: int = 0
    reentry_cooldown_minutes: int = 30
    
    timestamp: str = ""
    confidence_score: float = 0.0

class StrategyOptimizer:
    """Advanced strategy optimizer for trading bot"""
    
    def __init__(self, db_path: str = "trading_bot.db", strategy_func: Optional[Callable] = None):
        self.db_path = db_path
        self.current_market_condition = MarketCondition()
        self.market_regime = "RANGING"  # Default to ranging market

        if strategy_func:
            self.strategy_func = strategy_func
        else:
            # Define a default strategy if none is provided, maintaining original behavior
            def _default_rsi_strategy(df: pd.DataFrame, params: OptimizedParameters) -> pd.Series:
                delta = df['price'].diff()
                gain = delta.where(delta > 0, 0).ewm(com=13, min_periods=14).mean()
                loss = -delta.where(delta < 0, 0).ewm(com=13, min_periods=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                signals = pd.Series(0, index=df.index)
                signals.loc[df['rsi'] < params.rsi_oversold] = 1
                signals.loc[df['rsi'] > params.rsi_overbought] = -1
                return signals
            self.strategy_func = _default_rsi_strategy

        self.optimized_params = OptimizedParameters()
        self.base_params = OptimizedParameters()  # Fallback parameters
        
        # Performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'avg_duration': 0.0,
            'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'win_rate': 0.0
        })
        
        # Market condition history
        self.market_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        
        # Optimization settings
        self.optimization_interval = 3600  # 1 hour
        self.min_trades_for_optimization = 10
        self.confidence_threshold = 0.5
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'volume_threshold': (1.2, 3.0),
            'momentum_threshold': (0.01, 0.05),
            'position_size_multiplier': (0.5, 2.0),
            'stop_loss_multiplier': (0.5, 2.0),
            'take_profit_multiplier': (1.0, 3.0)
        }
        
        self.regime_thresholds = {
            'trend_strength': 0.03,
            'volatility': 0.03
        }
        
        self.optimization_active = True
        self.last_optimization = datetime.now()
        self._init_database()
        
    def _init_database(self):
        """Initialize optimizer database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        market_condition TEXT NOT NULL,
                        performance_metrics TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        old_parameters TEXT NOT NULL,
                        new_parameters TEXT NOT NULL,
                        optimization_reason TEXT NOT NULL,
                        confidence_score REAL NOT NULL
                    )
                """)
                
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error initializing optimizer database: {e}")
    
    def analyze_market_condition(self, market_data: Dict[str, Any]) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            prices = market_data.get('prices', [])
            if not prices:
                return MarketCondition()
            
            # Calculate basic market metrics
            volatility = calculate_volatility(prices) if len(prices) > 1 else 0.0
            
            # Determine trend strength
            if len(prices) >= 20:
                short_ma = sum(prices[-10:]) / 10
                long_ma = sum(prices[-20:]) / 20
                trend_strength = abs(short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
            else:
                trend_strength = 0.0
            
            return MarketCondition(
                volatility=volatility,
                trend_strength=trend_strength,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error analyzing market condition: {e}")
            return MarketCondition()
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator using an Exponential Moving Average."""
        if len(prices) < period + 1:
            return 50.0
        
        prices_series = pd.Series(prices)
        delta = prices_series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use exponential moving average for a more standard RSI calculation
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Get the last value
        last_avg_gain = avg_gain.iloc[-1]
        last_avg_loss = avg_loss.iloc[-1]

        if last_avg_loss == 0:
            return 100.0
        
        rs = last_avg_gain / last_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi if not np.isnan(rsi) else 50.0
    
    def optimize_parameters(
        self, performance_metrics: Dict[str, Any], market_data: Dict[str, Any]
    ) -> OptimizedParameters:
        """
        Optimize trading parameters using real-time performance and market conditions.
        
        Args:
            performance_metrics: A dictionary of real-time metrics, ideally from the MonitoringDashboard.
            market_data: A dictionary containing recent price and volume data.
        """
        try:
            # Log current performance before any optimization attempt
            self._log_strategy_performance(performance_metrics)

            # Check if optimization is needed
            if not self._should_optimize(performance_metrics):
                return self.optimized_params

            # 1. Start with a copy of the current parameters
            new_params = copy.deepcopy(self.optimized_params)

            # 2. Find the best parameters from similar historical conditions
            best_historical_params = self._find_best_historical_params()
            if best_historical_params:
                new_params = best_historical_params

            # 3. Run advanced optimization algorithms to refine further
            new_params = self._run_advanced_optimization(performance_metrics, market_data, new_params)

            # Calculate confidence score
            total_trades = performance_metrics.get('total_trades', 0)
            confidence = self._calculate_confidence_score(total_trades, new_params)
            new_params.confidence_score = confidence

            # Apply new parameters if confidence is high enough
            if confidence >= self.confidence_threshold:
                old_params = asdict(self.optimized_params)
                self.optimized_params = new_params
                self.last_optimization = datetime.now()
                self._log_optimization(old_params, asdict(new_params), confidence)
                
                print(f"Parameters optimized with confidence {confidence:.2f}. New pos size multiplier: {new_params.position_size_multiplier:.2f}")
            
            return self.optimized_params
            
        except Exception as e:
            print(f"Error optimizing parameters: {e}")
            return self.optimized_params
    
    def _should_optimize(self, performance_metrics: Dict[str, Any]) -> bool:
        """Determine if optimization should be performed"""
        # Check time since last optimization
        time_since_last = (datetime.now() - self.last_optimization).total_seconds()
        if time_since_last < self.optimization_interval:
            return False
        
        # Check if we have enough data
        total_trades = performance_metrics.get('total_trades', 0)
        if total_trades < self.min_trades_for_optimization:
            return False
        
        # Check if performance is declining (e.g., profit factor < 1.1)
        profit_factor = performance_metrics.get('profit_factor', 1.5)
        if profit_factor < 1.1:
            print(f"Triggering optimization due to low profit factor: {profit_factor:.2f}")
            return True
        
        # Check for high consecutive losses
        if performance_metrics.get('consecutive_losses', 0) > 4:
            print("Triggering optimization due to high consecutive losses.")
            return True
        
        return True
    
    def _generate_dynamic_parameters(self, performance_metrics: Dict[str, Any]) -> OptimizedParameters:
        """
        Generate optimized parameters by applying adjustment factors to base parameters.
        This approach is more data-driven and less reliant on hardcoded values.
        """
        # Start with a fresh copy of the base parameters
        params = copy.deepcopy(self.base_params)
        condition = self.current_market_condition
        
        # Initialize adjustment factors
        pos_size_factor, tp_factor, sl_factor = 1.0, 1.0, 1.0

        # Market Regime Adjustments
        if condition.regime == "TRENDING":
            tp_factor *= 1.2
            sl_factor *= 1.2
            params.rsi_overbought = 75.0
        else:  # RANGING
            tp_factor *= 0.9
            params.rsi_overbought = 68.0
            params.rsi_oversold = 32.0

        # Volatility Adjustments
        if condition.volatility > 0.05:
            pos_size_factor *= 0.75
        elif condition.volatility < 0.02:
            pos_size_factor *= 1.25

        # Sentiment Adjustments
        if condition.market_sentiment == "BULLISH":
            pos_size_factor *= 1.2
            tp_factor *= 1.2
        elif condition.market_sentiment == "BEARISH":
            pos_size_factor *= 0.8
            sl_factor *= 0.9

        # Performance-Based Adjustments
        win_rate = performance_metrics.get('win_rate', 0.5)
        profit_factor = performance_metrics.get('profit_factor', 1.5)
        consecutive_losses = performance_metrics.get('consecutive_losses', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0.0)

        if profit_factor < 1.2:
            pos_size_factor *= 0.9

        if max_drawdown > 0.1:
            pos_size_factor *= 0.8

        # Apply factors
        params.position_size_multiplier *= pos_size_factor
        params.take_profit_multiplier *= tp_factor
        params.stop_loss_multiplier *= sl_factor

        # Clamp parameters to safe ranges
        params.position_size_multiplier = np.clip(
            params.position_size_multiplier,
            self.param_ranges['position_size_multiplier'][0],
            self.param_ranges['position_size_multiplier'][1]
        )
        params.take_profit_multiplier = np.clip(
            params.take_profit_multiplier,
            self.param_ranges['take_profit_multiplier'][0],
            self.param_ranges['take_profit_multiplier'][1]
        )
        params.stop_loss_multiplier = np.clip(
            params.stop_loss_multiplier,
            self.param_ranges['stop_loss_multiplier'][0],
            self.param_ranges['stop_loss_multiplier'][1]
        )
        
        params.timestamp = datetime.now().isoformat()
        return params
    
    def _run_advanced_optimization(
        self, performance_metrics: Dict[str, Any], market_data: Dict[str, Any], seed_params: OptimizedParameters
    ) -> OptimizedParameters:
        """Run advanced optimization using a portfolio of algorithms."""
        try:
            # Get current win rate and trade history
            current_score = self._evaluate_parameter_set(self.optimized_params, market_data) # Use market_data
            prices = market_data.get('prices', [])
            
            if len(prices) < 100:
                print("Warning: Not enough historical data for advanced optimization. Skipping.")
                return self.optimized_params

            # Apply multiple optimization strategies
            genetic_params = self._genetic_algorithm_optimization(market_data, seed_params)
            random_search_params = self._random_search_optimization(market_data, seed_params)
            rule_based_params = self._generate_dynamic_parameters(performance_metrics)
            
            candidates = [
                ('genetic', genetic_params),
                ('random_search', random_search_params),
                ('rule_based', rule_based_params)
            ]
            
            best_params = self.optimized_params
            best_score = current_score
            
            for name, params in candidates:
                if params:
                    score = self._evaluate_parameter_set(params, market_data)
                    if score > best_score:
                        best_params = params
                        best_score = score
                        print(f"Optimization: {name} approach improved score to {score:.3f}")
            
            return best_params
            
        except Exception as e:
            print(f"Error in advanced optimization: {e}")
            return self.optimized_params
    
    def _genetic_algorithm_optimization(
        self, market_data: Dict[str, Any], seed_params: OptimizedParameters
    ) -> Optional[OptimizedParameters]:
        """Genetic algorithm for parameter optimization"""
        try:
            population_size = 20
            generations = 10
            mutation_rate = 0.1
            
            # Initialize population
            population = []
            for _ in range(population_size):
                params = copy.deepcopy(self.base_params)
                # Mutate parameters randomly
                params.rsi_oversold = np.random.uniform(20, 40)
                params.rsi_overbought = np.random.uniform(60, 80)
                params.volume_threshold = np.random.uniform(1.2, 3.0)
                params.position_size_multiplier = np.random.uniform(0.5, 2.0)
                params.stop_loss_multiplier = np.random.uniform(0.5, 2.0)
                params.take_profit_multiplier = np.random.uniform(1.0, 3.0)
                population.append(params)
            
            # Inject seed parameters
            population[0] = seed_params
            
            # Evolution loop
            for generation in range(generations):
                # Evaluate fitness
                fitness_scores = []
                for params in population:
                    score = self._evaluate_parameter_set(params, market_data)
                    fitness_scores.append(score)
                
                # Selection and crossover
                new_population = []
                for _ in range(population_size // 2):
                    # Tournament selection
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    child = self._crossover_parameters(parent1, parent2)
                    
                    # Mutation
                    if np.random.random() < mutation_rate:
                        child = self._mutate_parameters(child)
                    
                    new_population.extend([parent1, child]) # Keep one parent
                
                population = new_population
            
            # Return best individual
            final_scores = [self._evaluate_parameter_set(p, market_data) for p in population]
            best_idx = np.argmax(final_scores)
            return population[best_idx]
            
        except Exception as e:
            print(f"Error in genetic algorithm optimization: {e}")
            return None
    
    def _random_search_optimization(
        self, market_data: Dict[str, Any], seed_params: OptimizedParameters
    ) -> Optional[OptimizedParameters]:
        """Performs a guided random search for better parameters."""
        try:
            n_samples = 50
            best_params = seed_params
            best_score = self._evaluate_parameter_set(seed_params, market_data)
            
            # Sample parameter space
            for _ in range(n_samples):
                params = copy.deepcopy(self.base_params)
                
                # Sample from parameter ranges with Gaussian noise
                params.rsi_oversold = np.clip(np.random.normal(30, 5), 20, 40)
                params.rsi_overbought = np.clip(np.random.normal(70, 5), 60, 80)
                params.volume_threshold = np.clip(np.random.normal(1.8, 0.3), 1.2, 3.0)
                params.position_size_multiplier = np.clip(np.random.normal(1.0, 0.2), 0.5, 2.0)
                params.stop_loss_multiplier = np.clip(np.random.normal(1.0, 0.2), 0.5, 2.0)
                params.take_profit_multiplier = np.clip(np.random.normal(1.5, 0.3), 1.0, 3.0)
                
                # Evaluate
                score = self._evaluate_parameter_set(params, market_data)
                if score > best_score:
                    best_score = score
                    best_params = params
            
            return best_params
            
        except Exception as e:
            print(f"Error in random search optimization: {e}")
            return None
    
    def _evaluate_parameter_set(self, params: OptimizedParameters, market_data: Dict[str, Any]) -> float:
        """
        Evaluate a parameter set by running a mini-backtest.
        This is a simplified backtest simulation using the configured strategy.
        """
        prices = market_data.get('prices', [])
        if len(prices) < 50:
            return 0.0

        try:
            df = pd.DataFrame({'price': prices})
            df['signal'] = self.strategy_func(df, params)

            # Simulate Trades
            position = 0
            trades = []
            entry_price = 0
            
            stop_loss_pct = 0.05 * params.stop_loss_multiplier
            take_profit_pct = 0.10 * params.take_profit_multiplier

            for i, row in df.iterrows():
                if position == 0 and row['signal'] == 1:
                    position = 1
                    entry_price = row['price']
                elif position == 1:
                    pnl_pct = (row['price'] - entry_price) / entry_price
                    if row['signal'] == -1 or pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                        trades.append(pnl_pct)
                        position = 0

            if not trades:
                return 0.0

            # Calculate Performance
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            gross_profit = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            # Composite Score
            score = (
                (win_rate * 0.5) + (min(profit_factor, 3.0) / 3.0 * 0.5)
            )
            return score if not np.isnan(score) else 0.0
            
        except Exception as e:
            print(f"Error evaluating parameter set: {e}")
            return 0.0
    
    def _tournament_selection(self, population: List[OptimizedParameters], fitness_scores: List[float]) -> OptimizedParameters:
        """Tournament selection for the genetic algorithm."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover_parameters(self, parent1: OptimizedParameters, parent2: OptimizedParameters) -> OptimizedParameters:
        """Crossover (breeding) operation for the genetic algorithm."""
        child = copy.deepcopy(parent1)
        
        # Random crossover of parameters
        if np.random.random() < 0.5:
            child.rsi_oversold = parent2.rsi_oversold
        if np.random.random() < 0.5:
            child.rsi_overbought = parent2.rsi_overbought
        if np.random.random() < 0.5:
            child.volume_threshold = parent2.volume_threshold
        if np.random.random() < 0.5:
            child.position_size_multiplier = parent2.position_size_multiplier
        if np.random.random() < 0.5:
            child.stop_loss_multiplier = parent2.stop_loss_multiplier
        if np.random.random() < 0.5:
            child.take_profit_multiplier = parent2.take_profit_multiplier
        
        return child
    
    def _mutate_parameters(self, params: OptimizedParameters) -> OptimizedParameters:
        """Mutation operation for the genetic algorithm."""
        mutated = copy.deepcopy(params)
        
        # Small random mutations
        mutated.rsi_oversold += np.random.normal(0, 2)
        mutated.rsi_overbought += np.random.normal(0, 2)
        mutated.volume_threshold += np.random.normal(0, 0.1)
        mutated.position_size_multiplier += np.random.normal(0, 0.05)
        mutated.stop_loss_multiplier += np.random.normal(0, 0.05)
        mutated.take_profit_multiplier += np.random.normal(0, 0.1)
        
        # Clamp to valid ranges
        mutated.rsi_oversold = np.clip(mutated.rsi_oversold, 20, 40)
        mutated.rsi_overbought = np.clip(mutated.rsi_overbought, 60, 80)
        mutated.volume_threshold = np.clip(mutated.volume_threshold, 1.2, 3.0)
        mutated.position_size_multiplier = np.clip(mutated.position_size_multiplier, 0.5, 2.0)
        mutated.stop_loss_multiplier = np.clip(mutated.stop_loss_multiplier, 0.5, 2.0)
        mutated.take_profit_multiplier = np.clip(mutated.take_profit_multiplier, 1.0, 3.0)
        
        return mutated
    
    def _log_strategy_performance(self, performance_metrics: Dict[str, Any]):
        """Log the performance of the current strategy parameters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_performance
                    (strategy_name, parameters, market_condition, performance_metrics)
                    VALUES (?, ?, ?, ?)
                """, (
                    "main_strategy",
                    json.dumps(asdict(self.optimized_params)),
                    json.dumps(asdict(self.current_market_condition)),
                    json.dumps(performance_metrics)
                ))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging strategy performance: {e}")

    def _calculate_confidence_score(self, total_trades: int, new_params: OptimizedParameters) -> float:
        """Calculate confidence score for new parameters"""
        # Base confidence on data quality (number of trades) and market stability
        data_quality = min(1.0, total_trades / 100.0)  # Full confidence after 100 trades
        
        # Market stability (lower volatility = higher confidence)
        market_stability = max(0.0, 1.0 - self.current_market_condition.volatility * 10)
        
        # Combine factors
        confidence = (data_quality * 0.6 + market_stability * 0.4)
        return min(1.0, max(0.0, confidence))
    
    def _log_optimization(self, old_params: Dict, new_params: Dict, confidence: float):
        """Log optimization to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO optimization_history 
                    (old_parameters, new_parameters, optimization_reason, confidence_score)
                    VALUES (?, ?, ?, ?)
                """, (
                    json.dumps(old_params),
                    json.dumps(new_params),
                    f"Regime: {self.current_market_condition.regime}, Market: {self.current_market_condition.market_sentiment}, Vol: {self.current_market_condition.volatility:.3f}",
                    confidence
                ))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging optimization: {e}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimized parameters"""
        return asdict(self.optimized_params)
    
    def get_market_analysis(self) -> Dict[str, Any]:
        """Get current market analysis"""
        return {
            'current_condition': asdict(self.current_market_condition),
            'optimization_confidence': self.optimized_params.confidence_score,
            'last_optimization': self.last_optimization.isoformat(),
            'next_optimization': (self.last_optimization + timedelta(seconds=self.optimization_interval)).isoformat()
        }

    def _get_historical_performance_data(self) -> List[Dict[str, Any]]:
        """Retrieve historical performance and parameter data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Get data from the last 30 days
                thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute(
                    """
                    SELECT parameters, market_condition, performance_metrics
                    FROM strategy_performance WHERE timestamp >= ?
                    """,
                    (thirty_days_ago,)
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving historical performance data: {e}")
            return []

    def _find_best_historical_params(self) -> Optional[OptimizedParameters]:
        """Find the best performing parameters from similar historical market conditions."""
        historical_data = self._get_historical_performance_data()
        if not historical_data:
            return None

        current_cond = self.current_market_condition
        # Normalize current market features for comparison
        current_features = np.array([current_cond.volatility, current_cond.trend_strength])

        best_params = None
        highest_score = -1

        for record in historical_data:
            try:
                hist_cond = MarketCondition(**json.loads(record['market_condition']))
                hist_features = np.array([hist_cond.volatility, hist_cond.trend_strength])
                
                # Find conditions with low distance (i.e., similar)
                distance = np.linalg.norm(current_features - hist_features)
                if distance < 0.02:  # Threshold for "similar"
                    perf = json.loads(record['performance_metrics'])
                    # Use a simple score: profit_factor * win_rate
                    score = perf.get('profit_factor', 0) * perf.get('win_rate', 0)
                    if score > highest_score:
                        highest_score = score
                        best_params = OptimizedParameters(**json.loads(record['parameters']))
            except (json.JSONDecodeError, TypeError):
                continue
        
        return best_params
    
    def reset_to_defaults(self):
        """Reset parameters to default values"""
        self.optimized_params = OptimizedParameters()
        self.last_optimization = datetime.now()
        print("Parameters reset to defaults")

# Global optimizer instance
strategy_optimizer = StrategyOptimizer()

def get_strategy_optimizer():
    """Get the global strategy optimizer instance"""
    return strategy_optimizer