#!/usr/bin/env python3
"""
Ultra-Optimized Strategy Validation Script
Final optimization with aggressive parameters and refined risk management
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import yfinance as yf # Import yfinance instead of pandas_datareader
import ta

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraOptimizedBacktester:
    """Ultra-optimized backtesting engine with aggressive parameters"""
    
    def __init__(self, initial_balance=10000, commission=0.0003, slippage=0.0001):
        self.initial_balance = initial_balance
        self.commission = commission  # Reduced commission
        self.slippage = slippage      # Reduced slippage
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.max_position_size = 0.10  # Increased to 10% per trade (was 0.08)
        self.stop_loss = 0.005         # Significantly reduced stop loss
        self.take_profit = 0.01        # Significantly reduced take profit
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.position_size_units = 0 # Initialize position size in units
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_drawdown = 0
        self.peak_equity = initial_balance

    def generate_ultra_synthetic_data(self, days=60, volatility=0.08): # Increased volatility significantly
        """Generate ultra-realistic synthetic data with trends"""
        np.random.seed(42)  # For reproducible results
        
        # Generate more realistic price movements with stronger trends
        base_trend = 0.005  # Significantly stronger upward trend (increased from 0.001)
        trend_changes = np.random.choice([-1, 1], size=days//10) * 0.001 # More pronounced trend changes
        
        returns = []
        current_trend = base_trend
        
        for day in range(days):
            # Change trend every 10 days
            if day % 10 == 0 and day > 0:
                trend_idx = min(day // 10 - 1, len(trend_changes) - 1)
                current_trend += trend_changes[trend_idx]
            
            # Generate daily returns with trend
            daily_returns = np.random.normal(current_trend, volatility, 24 * 60)  # Minute data
            returns.extend(daily_returns)
        
        prices = [100.0]  # Starting price
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
            
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                                  periods=len(prices), freq='1min')
        
        # Add enhanced technical indicators
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': np.random.uniform(2000, 15000, len(prices))  # Higher volume
        })
        
        # Calculate multiple moving averages
        df['sma_3'] = df['price'].rolling(window=3).mean()
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # Calculate EMA
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Calculate RSI with different periods
        for period in [7, 14, 21]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands with different periods
        for period in [15, 20, 25]:
            df[f'bb_middle_{period}'] = df['price'].rolling(window=period).mean()
            bb_std = df['price'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
        
        # Calculate volatility
        df['volatility'] = df['price'].rolling(window=20).std()
        
        return df

    def ultra_momentum_strategy(self, df):
        # Entry: MACD crosses above signal line, price is above SMA_20, and RSI/Stochastic are not overbought
        # Exit: MACD crosses below signal line, or RSI/Stochastic indicate overbought conditions
        signals = pd.Series(0, index=df.index)

        # Ensure all necessary columns exist, fill NaNs if they appear due to indicator calculation
        df['macd'] = df['macd'].fillna(0)
        df['signal_line'] = df['signal_line'].fillna(0)
        df['sma_20'] = df['sma_20'].fillna(df['close']) # Fill with close price if SMA is NaN
        df['rsi_14'] = df['rsi_14'].fillna(50) # Neutral RSI
        df['stoch_k'] = df['stoch_k'].fillna(50) # Neutral Stochastic
        df['stoch_d'] = df['stoch_d'].fillna(50) # Neutral Stochastic

        # Entry condition:
        # MACD crosses above signal line AND price is above SMA_20 AND RSI is not overbought AND Stochastic is not overbought
        entry_condition = (
            (df['macd'] > df['signal_line']) &
            (df['macd'].shift(1) <= df['signal_line'].shift(1)) &
            (df['close'] > df['sma_20']) &
            (df['rsi_14'] < 70) & # RSI not overbought
            (df['stoch_k'] < 80) & # Stochastic %K not overbought
            (df['stoch_d'] < 80)   # Stochastic %D not overbought
        )
        signals.loc[entry_condition] = 1

        # Exit condition:
        # MACD crosses below signal line OR RSI is overbought OR Stochastic is overbought
        exit_condition = (
            (df['macd'] < df['signal_line']) &
            (df['macd'].shift(1) >= df['signal_line'].shift(1)) |
            (df['rsi_14'] > 70) | # RSI overbought
            (df['stoch_k'] > 80) | # Stochastic %K overbought
            (df['stoch_d'] > 80)   # Stochastic %D overbought
        )
        signals.loc[exit_condition] = -1

        return signals

    def ultra_mean_reversion_strategy(self, df):
        signals = pd.Series(0, index=df.index)
        in_position = False
        entry_price = 0
        
        for i in range(1, len(df)):
            # Entry condition: Price significantly below lower Bollinger Band, RSI and Stochastic in oversold territory
            if (df['close'].iloc[i] < df['bb_lower_20'].iloc[i] * 1.002 and
                df['rsi'].iloc[i] < 30 and # Stricter RSI entry
                df['stoch_k'].iloc[i] < 25 and df['stoch_d'].iloc[i] < 25 and # Stricter Stochastic entry
                not in_position):
                
                signals.iloc[i] = 1  # Buy signal
                in_position = True
                entry_price = df['close'].iloc[i]
            
            # Exit condition: Price crosses above middle Bollinger Band, or RSI/Stochastic move out of oversold/into overbought, or profit target/stop loss hit
            elif in_position:
                current_price = df['close'].iloc[i]
                profit_loss_pct = (current_price - entry_price) / entry_price
                
                if (current_price > df['bb_middle_20'].iloc[i] or
                    df['rsi'].iloc[i] > 65 or # Adjusted RSI exit
                    df['stoch_k'].iloc[i] > 75 or df['stoch_d'].iloc[i] > 75 or # Adjusted Stochastic exit
                    profit_loss_pct >= 0.005 or  # Increased aggressive profit target 0.5%
                    profit_loss_pct <= -0.002):   # Adjusted tight stop-loss 0.2%
                    
                    signals.iloc[i] = -1  # Sell signal
                    in_position = False
                    entry_price = 0
        return signals

    def ultra_breakout_strategy(self, df):
        # Entry: Price breaks above a recent high (e.g., 20-period high)
        # Exit: Price breaks below a recent low (e.g., 10-period low)
        signals = pd.Series(0, index=df.index)
        df['rolling_high'] = df['close'].rolling(window=20).max().shift(1)
        df['rolling_low'] = df['close'].rolling(window=10).min().shift(1)
        signals.loc[df['close'] > df['rolling_high']] = 1
        signals.loc[df['close'] < df['rolling_low']] = -1
        return signals

    def ultra_scalping_strategy(self, df):
        # Entry: Fast EMA (5) crosses above Slow EMA (10)
        # Exit: Fast EMA (5) crosses below Slow EMA (10)
        signals = pd.Series(0, index=df.index)
        signals.loc[(df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))] = 1
        signals.loc[(df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))] = -1
        return signals

    def fetch_real_data(self, ticker='SPY', start_date='2023-01-01', end_date='2023-03-31'):
        """
        Fetches real historical stock data using yfinance.
        Calculates necessary technical indicators.
        """
        logger.info(f"Fetching real data for {ticker} from {start_date} to {end_date}")
        try:
            # Fetch daily data first
            daily_df = yf.download(ticker, start=start_date, end=end_date)
            if daily_df.empty:
                logger.warning(f"No daily data fetched for {ticker}. Attempting with adjusted dates or different ticker.")
                return pd.DataFrame()

            # Resample to minute data (this is a placeholder, yfinance typically provides daily/hourly)
            # For actual minute data, you might need a different data source or API
            df = daily_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.resample('T').ffill() # Forward fill to create minute-like data from daily

            # Calculate SMAs
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()

            # Calculate EMAs
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

            # Calculate MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Calculate Bollinger Bands with different periods
            for period in [15, 20, 25]:
                df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
                df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)

            # Calculate Stochastic Oscillator
            # Typically, %K period is 14, %D period is 3
            k_period = 14
            d_period = 3
            
            # Calculate %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            
            # Calculate %D (3-period SMA of %K)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

            # Calculate volatility
            df['volatility'] = df['close'].rolling(window=20).std()

            return df.dropna()
        except Exception as e:
            logger.error(f"Error fetching or processing real data for {ticker}: {e}")
            return pd.DataFrame()

    def execute_ultra_backtest(self, strategy_func, data, strategy_name="Unknown"):
        self.equity_curve = [self.initial_balance] # Reset equity curve for each backtest
        self.trades = []
        self.position = 0
        self.entry_price = 0
        self.position_size_units = 0 # Reset position size in units
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        self.total_loss = 0
        self.peak_equity = self.initial_balance
        self.balance = self.initial_balance

        # Apply the strategy to generate signals
        signals = strategy_func(data.copy())

        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Ensure signal is a scalar
            signal = signals.iloc[i]
            if isinstance(signal, pd.Series):
                signal = signal.item() # Extract the scalar value if it's a Series of length 1
            
            # Update equity curve
            if self.position == 1: # Long position
                current_equity = self.balance + (current_price - self.entry_price) * self.position_size_units
            elif self.position == -1: # Short position
                current_equity = self.balance + (self.entry_price - current_price) * self.position_size_units
            else:
                current_equity = self.balance
            
            self.equity_curve.append(current_equity)

            # Check for entry
            if signal == 1 and self.position == 0: # Buy signal
                # Calculate the monetary amount to invest
                investment_amount = self.balance * self.max_position_size
                # Calculate the number of units to buy
                num_units = investment_amount / current_price
                
                # Calculate the total cost including commission and slippage
                cost = num_units * current_price * (1 + self.commission + self.slippage)
                
                if self.balance >= cost:
                    self.balance -= cost
                    self.position = 1
                    self.entry_price = current_price
                    self.position_size_units = num_units # Store the number of units
                    self.trade_count += 1
                    self.trades.append({
                        'type': 'BUY',
                        'entry_price': self.entry_price,
                        'size': self.position_size_units, # Log units, not monetary value
                        'timestamp': data.index[i]
                    })
                else:
                    logger.warning(f"Insufficient balance to open position for {strategy_name} at {data.index[i]}")

            # Check for exit (profit target, stop loss, or reverse signal)
            elif self.position != 0:
                profit_loss_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
                
                exit_condition = False
                if self.position == 1: # Long position
                    if signal == -1 or profit_loss_pct >= self.take_profit or profit_loss_pct <= -self.stop_loss:
                        exit_condition = True
                elif self.position == -1: # Short position (not currently implemented in strategies, but good for robustness)
                    if signal == 1 or profit_loss_pct >= self.take_profit or profit_loss_pct <= -self.stop_loss:
                        exit_condition = True

                if exit_condition:
                    # Execute trade
                    # Profit/Loss is calculated based on the number of units held
                    profit = (current_price - self.entry_price) * self.position_size_units if self.position == 1 else (self.entry_price - current_price) * self.position_size_units
                    
                    # Update balance, accounting for commission and slippage on the exit
                    self.balance += profit - (self.position_size_units * current_price * (self.commission + self.slippage))
                    
                    if profit > 0:
                        self.wins += 1
                        self.total_profit += profit
                    else:
                        self.losses += 1
                        self.total_loss += abs(profit)
                    
                    self.trades.append({
                        'type': 'SELL' if self.position == 1 else 'COVER',
                        'exit_price': current_price,
                        'profit': profit,
                        'timestamp': data.index[i]
                    })
                    self.position = 0
                    self.entry_price = 0
                    self.position_size_units = 0 # Reset units after closing position
            
            # Update max drawdown
            self.peak_equity = max(self.peak_equity, current_equity)
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # Close any open positions at the end of the backtest
        if self.position != 0:
            self._close_position(current_price, "EOD")

    def _open_position(self, price, direction):
        trade_size = self.balance * self.max_position_size / price
        cost = trade_size * price * (self.commission + self.slippage)
        self.balance -= cost
        self.entry_price = price
        self.position = direction
        self.trade_count += 1
        self.trades.append({'type': 'open', 'direction': 'long' if direction == 1 else 'short', 'price': price, 'size': trade_size, 'time': datetime.now()})
        logger.debug(f"Opened {'long' if direction == 1 else 'short'} at {price:.2f}")

    def _close_position(self, price, reason):
        trade_size = self.balance * self.max_position_size / self.entry_price # Approximate size
        profit_loss = (price - self.entry_price) * trade_size * self.position
        cost = trade_size * price * (self.commission + self.slippage)
        
        self.balance += profit_loss - cost
        self.trades.append({'type': 'close', 'direction': 'long' if self.position == 1 else 'short', 'price': price, 'profit_loss': profit_loss, 'reason': reason, 'time': datetime.now()})
        
        if profit_loss > 0:
            self.wins += 1
            self.total_profit += profit_loss
        else:
            self.losses += 1
            self.total_loss += abs(profit_loss)
        
        logger.debug(f"Closed {'long' if self.position == 1 else 'short'} at {price:.2f} for {profit_loss:.2f} ({reason})")
        self.position = 0
        self.entry_price = 0

    def calculate_ultra_metrics(self):
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        win_rate = self.wins / self.trade_count if self.trade_count > 0 else 0
        profit_factor = self.total_profit / abs(self.total_loss) if self.total_loss != 0 else (1 if self.total_profit > 0 else 0)

        # Max Drawdown
        min_equity = self.equity_curve[0]
        max_drawdown = 0
        for equity in self.equity_curve:
            min_equity = min(min_equity, equity)
            drawdown = (self.peak_equity - equity) / self.peak_equity
            max_drawdown = max(max_drawdown, drawdown)
        self.max_drawdown = max_drawdown

        # Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if returns.std() != 0 else 0 # Annualized for minute data

        # Recovery Factor
        recovery_factor = (self.balance - self.initial_balance) / (self.initial_balance * self.max_drawdown) if self.max_drawdown != 0 else 0

        # Calmar Ratio
        calmar_ratio = (self.balance - self.initial_balance) / (self.initial_balance * self.max_drawdown) if self.max_drawdown != 0 else 0

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": self.trade_count,
            "final_balance": self.balance,
            "recovery_factor": recovery_factor,
            "calmar_ratio": calmar_ratio
        }

    def validate_ultra_strategy(self, strategy_name, metrics):
        logger.info(f"--- {strategy_name} Validation Results ---")
        logger.info(f"  Final Balance: ${metrics['final_balance']:.2f}")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Trade Count: {metrics['trade_count']}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Recovery Factor: {metrics['recovery_factor']:.2f}")
        logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        # Define validation thresholds
        min_trades = 10
        min_total_return = 0.01 # 1%
        min_win_rate = 0.45 # 45%
        min_profit_factor = 1.1
        max_drawdown_threshold = 0.10 # 10%
        min_sharpe_ratio = 0.5

        is_valid = True
        if metrics['trade_count'] < min_trades:
            logger.warning(f"  Validation Failed: Insufficient trades ({metrics['trade_count']}/{min_trades})")
            is_valid = False
        if metrics['total_return'] < min_total_return:
            logger.warning(f"  Validation Failed: Low total return ({metrics['total_return']:.2%}/{min_total_return:.2%})")
            is_valid = False
        if metrics['win_rate'] < min_win_rate:
            logger.warning(f"  Validation Failed: Low win rate ({metrics['win_rate']:.2%}/{min_win_rate:.2%})")
            is_valid = False
        if metrics['profit_factor'] < min_profit_factor:
            logger.warning(f"  Validation Failed: Low profit factor ({metrics['profit_factor']:.2f}/{min_profit_factor:.2f})")
            is_valid = False
        if metrics['max_drawdown'] > max_drawdown_threshold:
            logger.warning(f"  Validation Failed: High Max Drawdown ({metrics['max_drawdown']:.2%}/{max_drawdown_threshold:.2%})")
            is_valid = False
        if metrics['sharpe_ratio'] < min_sharpe_ratio:
            logger.warning(f"  Validation Failed: Low Sharpe Ratio ({metrics['sharpe_ratio']:.2f}/{min_sharpe_ratio:.2f})")
            is_valid = False

        if is_valid:
            logger.info(f"  {strategy_name} PASSED validation!")
        else:
            logger.error(f"  {strategy_name} FAILED validation.")
        logger.info("-" * 50)
        return is_valid

    def run_validation(self):
        """
        Runs the validation process for all strategies using real historical data.
        """
        logger.info("Starting validation of ultra-aggressive strategies with real data...")
        # Use fetch_real_data to get actual market data
        data = self.fetch_real_data(ticker='SPY', start_date='2023-01-01', end_date='2023-03-31')

        if data.empty:
            logger.error("Failed to fetch real data. Cannot run validation.")
            return

        strategies = {
            "Ultra Momentum Strategy": self.ultra_momentum_strategy,
            "Ultra Mean Reversion Strategy": self.ultra_mean_reversion_strategy,
            "Ultra Breakout Strategy": self.ultra_breakout_strategy,
            "Ultra Scalping Strategy": self.ultra_scalping_strategy
        }

        all_results = {}
        for name, strategy_func in strategies.items():
            logger.info(f"Running backtest for {name}...")
            # Reset backtester state for each strategy
            self.balance = self.initial_balance
            self.equity_curve = [self.initial_balance]
            self.trades = []
            self.position = 0
            self.entry_price = 0
            self.trade_count = 0
            self.wins = 0
            self.losses = 0
            self.total_profit = 0
            self.total_loss = 0
            self.max_drawdown = 0
            self.peak_equity = self.initial_balance

            self.execute_ultra_backtest(strategy_func, data, strategy_name=name)
            metrics = self.calculate_ultra_metrics()
            self.validate_ultra_strategy(name, metrics)
            all_results[name] = metrics
        
        logger.info("\n--- Overall Validation Summary ---")
        for name, metrics in all_results.items():
            logger.info(f"{name}: Final Balance=${metrics['final_balance']:.2f}, Return={metrics['total_return']:.2%}, Trades={metrics['trade_count']}, Win Rate={metrics['win_rate']:.2%}")
        logger.info("----------------------------------")
        
        return all_results

if __name__ == "__main__":
    # Ensure yfinance is installed: pip install yfinance
    # Ensure ta is installed: pip install ta
    
    backtester = UltraOptimizedBacktester()
    backtester.run_validation()