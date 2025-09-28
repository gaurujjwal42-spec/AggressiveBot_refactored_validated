import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DB_FILE = 'trading_bot.db'

@contextmanager
def db_transaction():
    """Context manager for a database transaction.
    Handles connection, cursor, commit, rollback, and closing.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        logging.debug(f"Database transaction started")
        yield cursor
        conn.commit()
        logging.debug(f"Database transaction committed successfully")
    except sqlite3.Error as e:
        logging.error(f"Database transaction failed: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                logging.debug(f"Database transaction rolled back")
            except Exception as rollback_error:
                logging.error(f"Failed to rollback transaction: {rollback_error}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in database transaction: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn:
            try:
                conn.close()
                logging.debug(f"Database connection closed")
            except Exception as close_error:
                logging.error(f"Error closing database connection: {close_error}")

@contextmanager
def db_query():
    """Context manager for a read-only database query.
    Handles connection, cursor, and closing.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        logging.debug(f"Database read connection opened")
        yield conn.cursor()
    except sqlite3.Error as e:
        logging.error(f"Database query failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Unexpected error in database query: {e}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                conn.close()
                logging.debug(f"Database read connection closed")
            except Exception as close_error:
                logging.error(f"Error closing database connection: {close_error}")

def init_database():
    """Initializes the database and creates tables if they don't exist."""
    logging.info("Initializing database...")
    try:
        with db_transaction() as cursor:
            # Using a flexible JSON-based approach for positions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    position_data TEXT NOT NULL
            )
        ''')
            
            # Create index on symbol for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)
            ''')
            
            # Create trades history table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    trade_data TEXT NOT NULL
                )
            ''')
            
            # Create index on trades timestamp for faster historical queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)
            ''')
            
            logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

        # Trade history table with a field for the decision snapshot
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                type TEXT NOT NULL,
                usdt_amount REAL NOT NULL,
                price REAL NOT NULL,
                token_amount REAL NOT NULL,
                pnl REAL,
                reason TEXT,
                timestamp TEXT NOT NULL,
                decision_snapshot TEXT
            )
        ''')
    logging.info("Database initialized successfully.")

def load_positions():
    """Loads all active positions from the database."""
    with db_query() as cursor:
        cursor.execute("SELECT id, position_data FROM positions")
        rows = cursor.fetchall()
    
    positions = {}
    for row in rows:
        positions[row['id']] = json.loads(row['position_data'])
        # Ensure timestamp is a datetime object after loading
        if 'timestamp' in positions[row['id']]:
             positions[row['id']]['timestamp'] = datetime.fromisoformat(positions[row['id']]['timestamp'])
    return positions

def save_position(position):
    """Saves or updates a single position in the database."""
    # Convert datetime to string for JSON serialization
    position_to_save = position.copy()
    if 'timestamp' in position_to_save and hasattr(position_to_save['timestamp'], 'isoformat'):
        position_to_save['timestamp'] = position_to_save['timestamp'].isoformat()

    with db_transaction() as cursor:
        cursor.execute(
            "INSERT OR REPLACE INTO positions (id, symbol, position_data) VALUES (?, ?, ?)",
            (position['id'], position['symbol'], json.dumps(position_to_save))
        )

def delete_position(position_id):
    """Deletes a position from the database."""
    with db_transaction() as cursor:
        cursor.execute("DELETE FROM positions WHERE id = ?", (position_id,))

def add_trade_to_history(trade_record):
    """Adds a completed trade to the history table."""
    with db_transaction() as cursor:
        cursor.execute(
            """INSERT INTO trade_history (position_id, symbol, type, usdt_amount, price, token_amount, pnl, reason, timestamp, decision_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_record['id'], trade_record['symbol'], trade_record['type'], trade_record['usdt_amount'],
             trade_record['price'], trade_record['token_amount'], trade_record.get('pnl'), trade_record.get('reason'),
             trade_record['timestamp'], json.dumps(trade_record.get('decision_snapshot', {})))
        )

def get_trade_history_paginated(page=1, limit=15):
    """Retrieves a paginated list of trades from the history."""
    offset = (page - 1) * limit
    with db_query() as cursor:
        cursor.execute("SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
        trades = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("SELECT COUNT(id) FROM trade_history")
        total = cursor.fetchone()[0]
    return trades, total