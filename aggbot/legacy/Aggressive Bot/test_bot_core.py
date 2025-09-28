import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys
from datetime import datetime

# Add project root to path to allow imports from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the modules from the bot
import main as trading_bot
from risk_manager import RiskManager
from strategy_optimizer import OptimizedParameters

@patch('main.get_monitoring_dashboard', return_value=MagicMock())
class TestTradingBotCore(unittest.TestCase):

    def setUp(self):
        """This method is called before each test to set up the environment."""
        self.mock_config = {
            "LIVE_TRADING_ENABLED": True,
            "WALLET_ADDRESS": "0x98B9A98C86c7B4658c9C0D6698B7b3cEf5469A1a",
            "PRIVATE_KEY": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "BUDGET_USDT": 1000,
            "TRADE_USDT": 100,
            "MAX_OPEN_TRADES": 5,
            "TARGET_CYCLES_PER_HOUR": 180,
            "USE_COINGECKO_API": True,
            "USE_BINANCE_API": True,
            "MIN_BNB_FOR_TRADING": 0.01,
            "ATR_STOP_LOSS_MULTIPLIER": 2.0,
            "TAKE_PROFIT_PCT": 20.0,
            "STOP_LOSS_PCT": -10.0,
            "TRAILING_STOP_ACTIVATION_PCT": 5.0,
            "TRAILING_STOP_PCT": 2.5
        }
        # Patch the open function to mock config loading
        self.mock_open_patcher = patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config)))
        self.mock_open_patcher.start()
        
        # Patch external dependencies
        self.patch_requests = patch('main.requests.get')
        self.mock_requests_get = self.patch_requests.start()
        
        self.patch_w3 = patch('main.Web3')
        self.mock_w3_class = self.patch_w3.start()
        self.mock_w3_instance = self.mock_w3_class.return_value
        self.mock_w3_instance.is_connected.return_value = True
        self.mock_w3_instance.eth.get_balance.return_value = self.mock_w3_instance.to_wei(1, 'ether')
        self.mock_w3_instance.to_wei.side_effect = lambda val, unit: int(val * (10**18))

        # Patch the global singletons to inject mocks
        self.patch_optimizer = patch('main.get_strategy_optimizer')
        self.mock_get_optimizer = self.patch_optimizer.start()
        self.mock_optimizer = self.mock_get_optimizer.return_value
        self.mock_optimizer.optimized_params = OptimizedParameters()
        
        # Reset global state before each test
        trading_bot.config = {}
        trading_bot.positions = {}
        trading_bot.risk_manager = None
        trading_bot.strategy_optimizer = self.mock_optimizer
        trading_bot.account = MagicMock()
        trading_bot.account.address = "0x123"

    def tearDown(self):
        """This method is called after each test to clean up."""
        self.mock_open_patcher.stop()
        self.patch_requests.stop()
        self.patch_w3.stop()
        self.patch_optimizer.stop()

    def test_startup_validation_success(self):
        """Test a successful startup validation."""
        with patch('main.fetch_pairs', return_value=[{'id': 'bitcoin'}]):
             self.assertTrue(trading_bot.load_configuration())
             self.assertTrue(trading_bot.validate_startup_requirements())

    def test_startup_validation_failure(self):
        """Test a failed startup validation due to live trading being disabled."""
        self.mock_config["LIVE_TRADING_ENABLED"] = False
        m = mock_open(read_data=json.dumps(self.mock_config))
        with patch('builtins.open', m):
            self.assertTrue(trading_bot.load_configuration())
            self.assertFalse(trading_bot.validate_startup_requirements())

    @patch('main.get_current_portfolio_value', return_value=1000)
    @patch('main.perform_safety_checks', return_value=True)
    def test_execute_buy_trade_success(self, mock_safety_checks, mock_portfolio, mock_monitoring):
        """Test a successful buy trade execution."""
        trading_bot.load_configuration()
        trading_bot.initialize_web3()
        trading_bot.risk_manager = RiskManager(trading_bot.config)
        
        # Mock the transaction receipt and contract calls
        mock_receipt = {'status': 1, 'gasUsed': 50000}
        self.mock_w3_instance.eth.wait_for_transaction_receipt.return_value = mock_receipt
        self.mock_w3_instance.eth.contract.return_value.functions.getAmountsOut.return_value.call.return_value = [100, 200]
        self.mock_w3_instance.eth.contract.return_value.functions.approve.return_value.estimate_gas.return_value = 30000
        self.mock_w3_instance.eth.contract.return_value.functions.swapExactTokensForTokens.return_value.estimate_gas.return_value = 150000

        pair = {
            'baseToken': {'address': '0xtoken', 'symbol': 'TKN'},
            'priceUsd': 10.0
        }
        
        result = trading_bot.execute_trade(pair, 'buy', 100, trading_bot.account, atr=0.5, confidence=0.8)
        
        self.assertIsNotNone(result)
        self.assertTrue(result['success'])
        self.assertEqual(result['receipt'], mock_receipt)
        self.mock_w3_instance.eth.send_raw_transaction.assert_called_once()

    @patch('main.get_current_portfolio_value', return_value=1000)
    @patch('main.perform_safety_checks', return_value=True)
    def test_execute_buy_trade_failure(self, mock_safety_checks, mock_portfolio, mock_monitoring):
        """Test a failed buy trade execution (receipt status 0)."""
        trading_bot.load_configuration()
        trading_bot.initialize_web3()
        trading_bot.risk_manager = RiskManager(trading_bot.config)
        
        # Mock a failed transaction receipt
        mock_receipt = {'status': 0, 'gasUsed': 50000}
        self.mock_w3_instance.eth.wait_for_transaction_receipt.return_value = mock_receipt
        self.mock_w3_instance.eth.contract.return_value.functions.getAmountsOut.return_value.call.return_value = [100, 200]
        self.mock_w3_instance.eth.contract.return_value.functions.approve.return_value.estimate_gas.return_value = 30000
        self.mock_w3_instance.eth.contract.return_value.functions.swapExactTokensForTokens.return_value.estimate_gas.return_value = 150000

        pair = {
            'baseToken': {'address': '0xtoken', 'symbol': 'TKN'},
            'priceUsd': 10.0
        }
        
        result = trading_bot.execute_trade(pair, 'buy', 100, trading_bot.account, atr=0.5, confidence=0.8)
        
        self.assertIsNotNone(result)
        self.assertFalse(result['success'])

    def test_check_positions_for_closure_take_profit(self, mock_monitoring):
        """Test the take-profit logic for closing a position."""
        trading_bot.load_configuration()
        
        pos_id = "TKN_123"
        trading_bot.positions = {
            pos_id: {
                'symbol': 'TKN',
                'usdt_invested': 100,
                'unrealized_pnl': 25, # 25% profit, should trigger 20% TP
                'token_address': '0xtoken'
            }
        }
        
        with patch('main._process_sell_trade') as mock_sell:
            trading_bot.check_positions_for_closure([], trading_bot.account)
            mock_sell.assert_called_once_with(pos_id, trading_bot.account, exit_reason='tp')

    def test_check_positions_for_closure_stop_loss(self, mock_monitoring):
        """Test the stop-loss logic for closing a position."""
        trading_bot.load_configuration()
        
        pos_id = "TKN_123"
        trading_bot.positions = {
            pos_id: {
                'symbol': 'TKN',
                'usdt_invested': 100,
                'unrealized_pnl': -11, # -11% loss, should trigger -10% SL
                'token_address': '0xtoken'
            }
        }
        
        with patch('main._process_sell_trade') as mock_sell:
            trading_bot.check_positions_for_closure([], trading_bot.account)
            mock_sell.assert_called_once_with(pos_id, trading_bot.account, exit_reason='sl')

    def test_check_positions_for_closure_trailing_stop(self, mock_monitoring):
        """Test the trailing stop-loss logic for closing a position."""
        trading_bot.load_configuration()
        
        pos_id = "TKN_123"
        trading_bot.positions = {
            pos_id: {
                'symbol': 'TKN',
                'usdt_invested': 100,
                  'entry_price': 10.0,
                'tokens_bought': 10.0,
                'token_address': '0xtoken',
                'peak_pnl_pct': 10.0 # Peak PNL was 10%, activating the 5% trailing stop
            }
        }
        
        # Mock market data where the price has dropped enough to trigger the TSL.
        # The TSL is 2.5%. Since peak was 10%, the stop is at 10% - 2.5% = 7.5%.
        # A price of 10.7 would result in a 7% PNL, which is below 7.5%.
        # We assume the bot's internal `update_positions_pnl` will calculate this.
        mock_pairs = [{
            'baseToken': {'symbol': 'TKN'},
            'priceUsd': "10.7" # Using string to mimic API response
        }]

        with patch('main._process_sell_trade') as mock_sell:
            trading_bot.check_positions_for_closure(mock_pairs, trading_bot.account)
            mock_sell.assert_called_once_with(pos_id, trading_bot.account, exit_reason='tsl')

    @patch('main.get_current_portfolio_value', return_value=1000)
    def test_risk_manager_rejects_trade(self, mock_portfolio, mock_monitoring):
        """Test that the risk manager can correctly reject an oversized trade."""
        trading_bot.load_configuration()
        trading_bot.risk_manager = RiskManager(trading_bot.config)
        
        # Make the risk manager reject trades by setting a very low max position size
        trading_bot.risk_manager.max_single_position_pct = 1 # 1% of portfolio
        
        pair = {'baseToken': {'address': '0xtoken', 'symbol': 'TKN'}, 'priceUsd': 10.0}
        
        # This trade is 10% of portfolio ($100 of $1000), so it should be rejected
        result = trading_bot.execute_trade(pair, 'buy', 100, trading_bot.account, atr=0.5, confidence=0.8)
        
        self.assertIsNone(result, "Trade should have been rejected by the risk manager but was not.")

    @patch('main.fetch_pairs', return_value=[{'id': 'bitcoin'}])
    @patch('main.check_bnb_balance', return_value=0.1)
    def test_verify_live_trading_readiness_success(self, mock_bnb, mock_fetch, mock_monitoring):
        """Test a successful live trading readiness check."""
        trading_bot.load_configuration()
        trading_bot.initialize_web3()
        trading_bot.risk_manager = RiskManager(trading_bot.config)
        self.assertTrue(trading_bot.verify_live_trading_readiness())

    @patch('main.fetch_pairs', return_value=[]) # No market data
    @patch('main.check_bnb_balance', return_value=0.1)
    def test_verify_live_trading_readiness_failure(self, mock_bnb, mock_fetch, mock_monitoring):
        """Test a failed live trading readiness check due to no market data."""
        trading_bot.load_configuration()
        trading_bot.initialize_web3()
        trading_bot.risk_manager = RiskManager(trading_bot.config)
        self.assertFalse(trading_bot.verify_live_trading_readiness())

    @patch('main.calculate_position_size')
    @patch('main.analyze_technicals')
    @patch('main.analyze_fundamentals')
    def test_find_trade_opportunities_success(self, mock_analyze_fundamentals, mock_analyze_technicals, mock_calculate_position_size, mock_monitoring):
        """Test that a valid opportunity is found and processed through the analysis pipeline."""
        trading_bot.load_configuration()
        trading_bot.temporary_blacklist = {}
        trading_bot.positions = {}
        
        mock_pair = {'baseToken': {'symbol': 'GOOD'}, 'volume': {'h24': 5000000}}
        mock_pairs = [mock_pair]
        
        # Mock the analysis pipeline stages
        mock_analyze_fundamentals.return_value = True
        mock_analyze_technicals.return_value = (4, 0.5, MagicMock(close=10.0)) # (confidence, atr, last_indicators)
        mock_calculate_position_size.return_value = 100.0
        
        self.mock_optimizer.analyze_market_condition.return_value = MagicMock()
        self.mock_optimizer.optimize_parameters.return_value = self.mock_optimizer.base_params
        
        opportunities = trading_bot.find_trade_opportunities(mock_pairs)
        
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]['pair'], mock_pair)
        self.assertEqual(opportunities[0]['trade_amount'], 100.0)
        
        mock_analyze_fundamentals.assert_called_once()
        mock_analyze_technicals.assert_called_once()
        mock_calculate_position_size.assert_called_once()

    @patch('main.calculate_position_size')
    @patch('main.analyze_technicals')
    @patch('main.analyze_fundamentals')
    def test_find_trade_opportunities_technical_fail(self, mock_analyze_fundamentals, mock_analyze_technicals, mock_calculate_position_size, mock_monitoring):
        """Test that a pair is correctly filtered out at the technical analysis stage."""
        trading_bot.load_configuration()
        mock_pair = {'baseToken': {'symbol': 'OKAY'}, 'volume': {'h24': 5000000}}
        mock_analyze_fundamentals.return_value = True
        mock_analyze_technicals.return_value = (2, 0.5, None) # Confidence score of 2 is below threshold of 3
        
        opportunities = trading_bot.find_trade_opportunities([mock_pair])
        
        self.assertEqual(len(opportunities), 0)
        mock_analyze_technicals.assert_called_once()
        mock_calculate_position_size.assert_not_called() # Should not proceed to sizing