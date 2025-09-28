import logging
import pandas as pd
from binance.client import Client

import api_handler
import config

def should_buy(symbol: str) -> bool:
    """
    Determines if we should buy based on a Simple Moving Average (SMA) crossover strategy.
    Returns True if the short-term SMA has just crossed above the long-term SMA.
    """
    try:
        # 1. Get historical data
        # We need more than LONG_SMA_PERIOD candles to ensure the SMA is accurate
        klines = api_handler.get_historical_data(
            symbol=symbol,
            interval=config.SMA_TIMEFRAME,
            limit=config.LONG_SMA_PERIOD + 5 
        )

        # 2. Create a pandas DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])

        # 3. Calculate SMAs
        df['short_sma'] = df['close'].rolling(window=config.SHORT_SMA_PERIOD).mean()
        df['long_sma'] = df['close'].rolling(window=config.LONG_SMA_PERIOD).mean()

        # 4. Check for crossover signal
        # The signal is a buy when the short SMA crosses ABOVE the long SMA.
        # We check the last two candles to confirm a recent crossover.
        previous_candle = df.iloc[-2]
        last_candle = df.iloc[-1]

        if previous_candle['short_sma'] < previous_candle['long_sma'] and last_candle['short_sma'] > last_candle['long_sma']:
            logging.info(f"[{symbol}] BUY SIGNAL: SMA Crossover detected. Short SMA ({last_candle['short_sma']:.2f}) crossed above Long SMA ({last_candle['long_sma']:.2f}).")
            return True

    except Exception as e:
        logging.error(f"[{symbol}] Could not execute strategy check: {e}", exc_info=True)
    
    return False