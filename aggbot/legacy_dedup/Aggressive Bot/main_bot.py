import time
import logging
import threading
from datetime import datetime, timezone

import database
import api_handler
import config # Import the new config
import strategy # Import our new strategy module
import trade_logger # Import the new trade logger module
from terminal_monitor import TerminalMonitor
from monitoring_dashboard import get_monitoring_dashboard
from diagnostic_error_handler import DiagnosticErrorHandler
from advanced_alert_system import get_alert_system, AlertType, AlertSeverity

def run_bot():
    """Main function to run the trading bot."""
    logging.info("--- Starting Multi-Symbol Trading Bot ---")
    database.init_database()
    
    # Initialize monitoring dashboard
    monitoring_dashboard = get_monitoring_dashboard()
    
    # Initialize trade analyzer
    from trade_analyzer import initialize_trade_analysis, get_trade_analyzer
    initial_portfolio_value = config.INITIAL_PORTFOLIO_VALUE # Assuming this is defined in config
    initialize_trade_analysis(initial_portfolio_value)
    trade_analyzer = get_trade_analyzer()
    logging.info("Trade analysis system initialized")
    
    # Initialize diagnostic error handler
    error_handler = DiagnosticErrorHandler(max_retries=3, base_delay=1.0)
    logging.info("Diagnostic error handler initialized")
    
    # Initialize advanced alert system
    alert_system = get_alert_system({
        'consecutive_failures': 5,
        'error_rate_per_minute': 10,
        'api_timeout_threshold': 30.0,
        'memory_usage_threshold': 85.0,
        'cpu_usage_threshold': 90.0
    })
    logging.info("Advanced alert system initialized")
    
    # Start terminal monitor in a separate thread
    terminal_monitor = TerminalMonitor()
    monitor_thread = threading.Thread(target=terminal_monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    logging.info("Terminal monitoring started. Press 'q' in the monitor window to quit.")
    
    # Initialize trade logger
    trade_logger.setup_logging()

    # --- Main Trading Loop ---
    while True:
        logging.info("--- Starting new trading cycle for all symbols ---")
        
        # Load all active positions from the database at the start of the cycle
        active_positions = database.load_positions()
        
        for symbol in config.SYMBOLS_TO_TRADE:
            try:
                # Find if we have a position for the current symbol in the loop
                position_for_symbol = next((pos for pos in active_positions.values() if pos['symbol'] == symbol), None)
                
                # Get current price with error handling
                current_price = error_handler.execute_with_diagnostics(
                    api_handler.get_current_price,
                    f"get_price_{symbol}",
                    {'symbol': symbol, 'operation': 'price_fetch'},
                    symbol
                )

                if position_for_symbol:
                    # --- LOGIC FOR WHEN A POSITION IS OPEN ---
                    entry_price = position_for_symbol['entry_price']
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100

                    logging.info(f"[{symbol}] Position found. Entry: ${entry_price:.2f}, Current: ${current_price:.2f}, PnL: {pnl_percent:.2f}%")

                    # Decision: Check for take-profit or stop-loss
                    if pnl_percent >= config.PROFIT_TARGET_PERCENT or pnl_percent <= -config.STOP_LOSS_PERCENT:
                        reason = "Take-Profit" if pnl_percent > 0 else "Stop-Loss"
                        logging.info(f"[{symbol}] Condition met: {reason}. Attempting to sell.")

                        try:
                            # Log detailed information before executing the trade
                            logging.info(f"[{symbol}] Executing SELL order with token amount: {position_for_symbol['token_amount']:.8f}")
                            
                            # Execute the sell order with error handling
                            sell_result = error_handler.execute_with_diagnostics(
                                api_handler.place_sell_order,
                                f"sell_order_{symbol}",
                                {
                                    'symbol': symbol, 
                                    'operation': 'sell_order',
                                    'amount': position_for_symbol['token_amount'],
                                    'reason': reason
                                },
                                symbol, 
                                position_for_symbol['token_amount']
                            )
                            
                            # Log the trade execution details
                            trade_logger.log_trade_execution(
                                action="SELL",
                                symbol=symbol,
                                amount=position_for_symbol['token_amount'],
                                price=sell_result['price'],
                                order_id=sell_result.get('id'),
                                status=sell_result.get('status')
                            )
                            
                            # Log detailed trade information to JSON and CSV
                            trade_data = {
                                **sell_result,
                                'symbol': symbol,
                                'type': 'SELL',
                                'token_amount': position_for_symbol['token_amount'],
                                'reason': reason,
                                'pnl': pnl_percent,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                            trade_logger.log_trade_to_json(trade_data)
                            trade_logger.log_trade_to_csv(trade_data)
                            
                            # Update monitoring dashboard with trade data
                            monitoring_dashboard.update_trade_metrics(trade_data)
                            
                            # Verify the sell order was executed successfully
                            if 'id' in sell_result:
                                verification = api_handler.verify_order_status(sell_result['id'], symbol)
                                trade_logger.log_trade_verification(symbol, sell_result['id'], verification)
                                
                                # Update sell_result with verification data if needed
                                if not verification['is_complete']:
                                    logging.warning(f"[{symbol}] SELL order not fully filled. Only {verification['fill_percentage']:.2f}% executed.")
                            else:
                                logging.warning(f"[{symbol}] Could not verify SELL order - no order ID returned")
                            
                            # Record the closed trade in history
                            trade_record = {
                                'id': position_for_symbol['id'],
                                'symbol': symbol,
                                'type': 'SELL',
                                'usdt_amount': sell_result['usdt_amount'],
                                'price': sell_result['price'],
                                'token_amount': position_for_symbol['token_amount'],
                                'pnl': pnl_percent,
                                'reason': reason,
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'decision_snapshot': {'entry_price': entry_price, 'close_price': sell_result['price']}
                            }
                            database.add_trade_to_history(trade_record)
                            database.delete_position(position_for_symbol['id'])
                            logging.info(f"[{symbol}] SELL successful. Position closed. PnL: {pnl_percent:.2f}%")
                            
                            # Analyze trading patterns for anomalies
                            alert_system.analyze_trading_patterns({
                                'symbol': symbol,
                                'type': 'SELL',
                                'amount': position_for_symbol['token_amount'],
                                'price': sell_result['price'],
                                'pnl': pnl_percent,
                                'execution_time': sell_result.get('execution_time', 0)
                            })
                        except Exception as e:
                            logging.error(f"[{symbol}] SELL order failed: {e}")
                            logging.error(f"[{symbol}] Position will remain open until next cycle.")
                            continue
                
                else:
                    # --- LOGIC FOR WHEN NO POSITION IS OPEN ---
                    logging.info(f"[{symbol}] No active position. Looking for an entry.")
                    
                    # Use the SMA Crossover strategy to decide whether to buy
                    if strategy.should_buy(symbol):
                        try:
                            # Log detailed information before executing the trade
                            logging.info(f"[{symbol}] Executing BUY order with USDT amount: ${config.USDT_TRADE_AMOUNT:.2f}")
                            
                            # Execute the buy order with error handling
                            buy_result = error_handler.execute_with_diagnostics(
                                api_handler.place_buy_order,
                                f"buy_order_{symbol}",
                                {
                                    'symbol': symbol,
                                    'operation': 'buy_order',
                                    'usdt_amount': config.USDT_TRADE_AMOUNT,
                                    'reason': 'strategy_signal'
                                },
                                symbol,
                                config.USDT_TRADE_AMOUNT
                            )
                            
                            # Log the trade execution details
                            trade_logger.log_trade_execution(
                                action="BUY",
                                symbol=symbol,
                                amount=config.USDT_TRADE_AMOUNT,
                                price=buy_result['price'],
                                order_id=buy_result.get('id'),
                                status=buy_result.get('status')
                            )
                            
                            # Log detailed trade information to JSON and CSV
                            trade_data = {
                                **buy_result,
                                'symbol': symbol,
                                'type': 'BUY',
                                'usdt_amount': config.USDT_TRADE_AMOUNT,
                                'reason': 'strategy_signal',
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                            trade_logger.log_trade_to_json(trade_data)
                            trade_logger.log_trade_to_csv(trade_data)
                            
                            # Update monitoring dashboard with trade data
                            monitoring_dashboard.update_trade_metrics(trade_data)
                            
                            # Verify the buy order was executed successfully
                            if 'id' in buy_result:
                                verification = api_handler.verify_order_status(buy_result['id'], symbol)
                                trade_logger.log_trade_verification(symbol, buy_result['id'], verification)
                                
                                # Update buy_result with verification data if needed
                                if not verification['is_complete']:
                                    logging.warning(f"[{symbol}] BUY order not fully filled. Only {verification['fill_percentage']:.2f}% executed.")
                            else:
                                logging.warning(f"[{symbol}] Could not verify BUY order - no order ID returned")

                            # Create and save the new position
                            new_position = {
                                'id': buy_result['id'],
                                'symbol': symbol,
                                'entry_price': buy_result['price'],
                                'usdt_amount': config.USDT_TRADE_AMOUNT,
                                'token_amount': buy_result['token_amount'],
                                'timestamp': datetime.now(timezone.utc).isoformat() # Save as string
                            }
                            database.save_position(new_position)
                            logging.info(f"[{symbol}] BUY successful. New position opened with ID {new_position['id']}. Amount: {buy_result['token_amount']} at ${buy_result['price']:.2f}")
                            
                            # Analyze trading patterns for anomalies
                            alert_system.analyze_trading_patterns({
                                'symbol': symbol,
                                'type': 'BUY',
                                'amount': buy_result['token_amount'],
                                'price': buy_result['price'],
                                'pnl': 0,  # New position, no PnL yet
                                'execution_time': buy_result.get('execution_time', 0)
                            })
                        except Exception as e:
                            logging.error(f"[{symbol}] BUY order failed: {e}")
                            continue

            except api_handler.APIError as e:
                # Handle API errors with diagnostics
                error_info = error_handler.handle_error_with_diagnostics(
                    e, 
                    f"api_operation_{symbol}", 
                    {'symbol': symbol, 'operation': 'trading_cycle'}
                )
                logging.error(f"[{symbol}] API Error: {error_info['user_message']}")
                
                # Record error for pattern analysis
                alert_system.record_error(e, "API_ERROR", error_info['diagnostic_info']['severity'], 
                                         {'symbol': symbol, 'operation': 'trading_cycle'})
            except Exception as e:
                # Handle unexpected errors with diagnostics
                error_info = error_handler.handle_error_with_diagnostics(
                    e, 
                    f"trading_cycle_{symbol}", 
                    {'symbol': symbol, 'operation': 'trading_cycle'}
                )
                logging.error(f"[{symbol}] Unexpected Error: {error_info['user_message']}")
                logging.debug(f"[{symbol}] Diagnostic info: {error_info['diagnostic_info']['diagnosis']}")
                logging.debug(f"[{symbol}] Suggested action: {error_info['diagnostic_info']['suggested_action']}")
                
                # Record error for pattern analysis
                alert_system.record_error(e, "SYSTEM_ERROR", error_info['diagnostic_info']['severity'], 
                                         {'symbol': symbol, 'operation': 'trading_cycle'})
        
        # Monitor system health and performance
        try:
            import psutil
            system_metrics = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'api_response_time': getattr(api_handler, 'last_response_time', 0)
            }
            alert_system.monitor_system_health(system_metrics)
        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            logging.warning(f"System health monitoring failed: {str(e)}")
        
        logging.info(f"--- Cycle finished. Waiting for {config.LOOP_INTERVAL_SECONDS} seconds. ---")
        time.sleep(config.LOOP_INTERVAL_SECONDS)