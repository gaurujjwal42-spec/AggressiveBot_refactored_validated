#!/usr/bin/env python3
"""
Machine Learning Model Training Script

This script is responsible for training the price prediction models.
It fetches historical data, performs feature engineering, trains the models,
and saves the trained artifacts for the main bot to use.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_FILENAME = 'ml_model.joblib'
FEATURES_FILENAME = 'ml_features.joblib'

# --- Feature Engineering ---
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators to be used as features."""
    df['returns'] = df['close'].pct_change()
    
    # SMA
    df['sma_15'] = df['close'].rolling(window=15).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    
    df.dropna(inplace=True)
    return df

def create_labels(df: pd.DataFrame, look_ahead_periods: int = 10, tp_pct: float = 0.02, sl_pct: float = 0.01) -> pd.DataFrame:
    """Creates target labels for classification."""
    df['future_price'] = df['close'].shift(-look_ahead_periods)
    df['price_change'] = (df['future_price'] - df['close']) / df['close']
    
    def get_label(change):
        if change > tp_pct:
            return 1  # Buy
        elif change < -sl_pct:
            return -1 # Sell
        else:
            return 0  # Hold
            
    df['label'] = df['price_change'].apply(get_label)
    df.dropna(inplace=True)
    return df

def train_model():
    """Main function to train and save the model."""
    logging.info("Starting model training process...")

    # 1. Load Data
    # In a real scenario, you would fetch this from an API (e.g., Binance)
    # For this example, we assume a 'historical_data.csv' file exists.
    try:
        # Make sure you have a CSV with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        df = pd.read_csv('historical_data.csv', parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        logging.info(f"Loaded {len(df)} rows of historical data.")
    except FileNotFoundError:
        logging.error("`historical_data.csv` not found. Please provide historical data for training.")
        return

    # 2. Feature Engineering
    logging.info("Preparing features...")
    df = prepare_features(df)

    # 3. Create Labels
    logging.info("Creating labels...")
    df = create_labels(df)

    if df.empty:
        logging.error("Not enough data to create features and labels. Aborting training.")
        return

    # 4. Define Features (X) and Target (y)
    features = [
        'returns', 'sma_15', 'sma_50', 'ema_12', 'ema_26', 
        'macd', 'macd_signal', 'rsi', 'bb_mid', 'bb_std'
    ]
    X = df[features]
    y = df['label']

    logging.info(f"Feature set contains {len(X)} samples.")
    logging.info(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Train Models
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    logging.info("Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    # 7. Evaluate Models
    logging.info("--- Random Forest Classification Report ---")
    rf_preds = rf_model.predict(X_test)
    logging.info("\n" + classification_report(y_test, rf_preds))

    logging.info("--- Gradient Boosting Classification Report ---")
    gb_preds = gb_model.predict(X_test)
    logging.info("\n" + classification_report(y_test, gb_preds))

    # 8. Save Artifacts
    logging.info(f"Saving models to `{MODEL_FILENAME}`...")
    joblib.dump({'rf': rf_model, 'gb': gb_model}, MODEL_FILENAME)

    logging.info(f"Saving feature list to `{FEATURES_FILENAME}`...")
    joblib.dump(features, FEATURES_FILENAME)

    logging.info("Model training complete and artifacts saved.")

if __name__ == '__main__':
    train_model()
