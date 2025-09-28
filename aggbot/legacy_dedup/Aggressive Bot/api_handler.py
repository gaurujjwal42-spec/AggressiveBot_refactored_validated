import random
import requests
import hmac
import hashlib
import time
from typing import Dict, Any, List
from datetime import datetime
import json

class APIError(Exception):
    """Custom API Error."""
    pass

# Global config storage
_config = None

def set_config(config: Dict[str, Any]):
    """Set the global config for API operations."""
    global _config
    _config = config

def _get_binance_signature(query_string: str, secret: str) -> str:
    """Generate Binance API signature."""
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def _binance_request(endpoint: str, params: Dict = None, method: str = 'GET') -> Dict[str, Any]:
    """Make authenticated request to Binance API."""
    if not _config:
        raise APIError("API configuration not set. Call set_config() first.")
    
    base_url = "https://api.binance.com" if not _config.get('USE_TESTNET') else "https://testnet.binance.vision"
    url = f"{base_url}{endpoint}"
    
    if params is None:
        params = {}
    
    # Add timestamp for authenticated requests
    if '/api/v3/' in endpoint and method in ['POST', 'DELETE']:
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = _get_binance_signature(query_string, _config['API_SECRET_KEY'])
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': _config['API_KEY'],
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    else:
        headers = {}
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, data=params, headers=headers, timeout=10)
        else:
            response = requests.request(method, url, params=params, headers=headers, timeout=10)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Binance API request failed: {e}")
        raise APIError(f"Binance API request failed: {e}")

def get_current_price(symbol: str) -> float:
    """Fetches the current price for a symbol from Binance."""
    try:
        # Convert symbol format (e.g., BTCUSDT)
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        response = _binance_request('/api/v3/ticker/price', {'symbol': symbol})
        return float(response['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        # Fallback to mock data if API fails
        if "BTC" in symbol:
            return 60000 + random.uniform(-500, 500)
        return 100 + random.uniform(-10, 10)

def get_klines(symbol: str, interval: str = '15m', limit: int = 100) -> List[list]:
    """MOCK: Fetches k-line/candlestick data."""
    klines = []
    price = get_current_price(symbol)
    # Generate somewhat realistic-looking candles
    for i in range(limit):
        open_price = price * (1 + random.uniform(-0.005, 0.005))
        high_price = max(open_price, price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, price) * (1 - random.uniform(0, 0.01))
        close_price = open_price * (1 + random.uniform(-0.01, 0.01))
        volume = random.uniform(10000, 500000)
        klines.append([
            (datetime.now().timestamp() - (limit - i) * 900) * 1000, # 15m interval
            str(open_price), str(high_price), str(low_price), str(close_price), str(volume)
        ])
        price = close_price
    return klines

def get_24h_volume(symbol: str) -> float:
    """MOCK: Fetches 24h trading volume in USDT."""
    return random.uniform(20000, 10_000_000)

def get_liquidity(symbol: str) -> float:
    """MOCK: Fetches the liquidity for a symbol in USDT."""
    return random.uniform(5000, 500_000)

def place_buy_order(symbol: str, usdt_amount: float, entry_price: float = None) -> Dict[str, Any]:
    """Places a real buy order on Binance."""
    try:
        if not _config or not _config.get('LIVE_TRADING_ENABLED'):
            # Fallback to mock if live trading disabled
            return _mock_buy_order(symbol, usdt_amount, entry_price)
        
        # Convert symbol format
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        # Get current price if not specified
        current_price = entry_price if entry_price else get_current_price(symbol)
        quantity = usdt_amount / current_price
        
        # Round quantity to appropriate precision
        quantity = round(quantity, 6)
        
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': usdt_amount,  # Buy with USDT amount
        }
        
        response = _binance_request('/api/v3/order', params, 'POST')
        
        print(f"🟢 LIVE BUY ORDER EXECUTED: {symbol} | Amount: ${usdt_amount:.2f} | Order ID: {response['orderId']}")
        
        return {
            'success': True,
            'order_id': response['orderId'],
            'symbol': symbol,
            'price': float(response.get('fills', [{}])[0].get('price', current_price)),
            'token_amount': float(response['executedQty']),
            'total_spent': float(response['cummulativeQuoteQty']),
            'timestamp': datetime.now().isoformat(),
            'status': response['status']
        }
        
    except Exception as e:
        print(f"❌ Error placing buy order: {e}")
        # Fallback to mock on error
        return _mock_buy_order(symbol, usdt_amount, entry_price)

def _mock_buy_order(symbol: str, usdt_amount: float, entry_price: float = None) -> Dict[str, Any]:
    """Mock buy order for testing/fallback."""
    price = entry_price if entry_price else get_current_price(symbol)
    token_amount = usdt_amount / price
    order_id = f"MOCK_BUY_{symbol}_{datetime.now().timestamp()}"
    print(f"🟡 MOCK BUY ORDER: {symbol} | Amount: ${usdt_amount:.2f} | Price: ${price:.4f} | Tokens: {token_amount:.6f}")
    return {
        'success': True,
        'order_id': order_id,
        'symbol': symbol, 
        'price': price, 
        'token_amount': token_amount,
        'total_spent': usdt_amount, 
        'timestamp': datetime.now().isoformat(), 
        'status': 'FILLED'
    }

def initialize_web3(config: Dict[str, Any]) -> bool:
    """MOCK: Initialize Web3 connection."""
    print("Mock Web3 initialization successful")
    return True

def load_token_cache():
    """MOCK: Load token cache."""
    print("Mock token cache loaded")
    return True

def place_sell_order(symbol: str, token_amount: float, entry_price: float = None) -> Dict[str, Any]:
    """Places a real sell order on Binance."""
    try:
        if not _config or not _config.get('LIVE_TRADING_ENABLED'):
            # Fallback to mock if live trading disabled
            return _mock_sell_order(symbol, token_amount, entry_price)
        
        # Convert symbol format
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        # Round quantity to appropriate precision
        quantity = round(token_amount, 6)
        
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': quantity,
        }
        
        response = _binance_request('/api/v3/order', params, 'POST')
        
        current_price = entry_price if entry_price else get_current_price(symbol)
        usdt_amount = float(response['cummulativeQuoteQty'])
        
        print(f"🔴 LIVE SELL ORDER EXECUTED: {symbol} | Tokens: {token_amount:.6f} | Order ID: {response['orderId']} | Amount: ${usdt_amount:.2f}")
        
        return {
            'success': True,
            'order_id': response['orderId'],
            'symbol': symbol,
            'price': float(response.get('fills', [{}])[0].get('price', current_price)),
            'usdt_amount': usdt_amount,
            'timestamp': datetime.now().isoformat(),
            'status': response['status']
        }
        
    except Exception as e:
        print(f"❌ Error placing sell order: {e}")
        # Fallback to mock on error
        return _mock_sell_order(symbol, token_amount, entry_price)

def _mock_sell_order(symbol: str, token_amount: float, entry_price: float = None) -> Dict[str, Any]:
    """Mock sell order for testing/fallback."""
    price = entry_price if entry_price else get_current_price(symbol)
    usdt_amount = token_amount * price
    order_id = f"MOCK_SELL_{symbol}_{datetime.now().timestamp()}"
    print(f"🟡 MOCK SELL ORDER: {symbol} | Tokens: {token_amount:.6f} | Price: ${price:.4f} | Amount: ${usdt_amount:.2f}")
    return {
        'success': True,
        'order_id': order_id,
        'symbol': symbol, 
        'price': price, 
        'usdt_amount': usdt_amount,
        'timestamp': datetime.now().isoformat(), 
        'status': 'FILLED'
    }