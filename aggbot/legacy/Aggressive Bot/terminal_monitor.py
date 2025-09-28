#!/usr/bin/env python3
"""
Real-time Terminal Monitoring for Trading Bot
Provides a command-line interface for monitoring trading activities
"""

import os
import sys
import time
import threading
import curses
import logging
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Optional

# Import local modules for data structures
from monitoring_dashboard import MonitoringDashboard, TradeMetrics, SystemMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('terminal_monitor')

class TerminalMonitor:
    """Terminal-based real-time monitoring for trading bot"""
    
    def __init__(self):
        self.monitoring_dashboard = MonitoringDashboard() # Local instance for data storage
        self.active_positions_count = 0
        self.recent_trades = deque(maxlen=20)
        self.recent_errors = deque(maxlen=10)
        self.recent_alerts = deque(maxlen=10)
        self.update_interval = 1.0  # seconds
        self.data_file = 'dashboard_data.json'
        self.running = False
        self.update_thread = None
        self.screen = None
        self.max_y = 0
        self.max_x = 0
        self.trade_window = None
        self.metrics_window = None
        self.alert_window = None
        self.log_window = None
        self.status_window = None
        self.command_window = None
        self.last_update_time = datetime.now()
        
    def start(self):
        """Start the terminal monitor"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_data)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Initialize curses
        curses.wrapper(self._main_loop)
    
    def stop(self):
        """Stop the terminal monitor"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
    
    def _update_data(self):
        """Update data in background thread"""
        while self.running:
            try:
                if os.path.exists(self.data_file):
                    with open(self.data_file, 'r') as f:
                        data = json.load(f)

                    # Update metrics
                    self.monitoring_dashboard.trade_metrics = TradeMetrics(**data.get('trade_metrics', {}))
                    self.monitoring_dashboard.system_metrics = SystemMetrics(**data.get('system_metrics', {}))

                    # Update active positions count
                    self.active_positions_count = data.get('system_metrics', {}).get('active_positions', 0)

                    # Update recent trades, errors, alerts
                    self.recent_trades = deque(data.get('recent_trades', []), maxlen=20)
                    self.recent_errors = deque(data.get('recent_errors', []), maxlen=10)

                    # Combine performance and risk alerts
                    alerts = data.get('performance_alerts', []) + data.get('risk_alerts', [])
                    self.recent_alerts = deque(sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True), maxlen=10)

                    self.last_update_time = datetime.now()

            except (json.JSONDecodeError, FileNotFoundError):
                # File might be being written or not created yet, just wait.
                time.sleep(self.update_interval)
                continue
            except Exception as e:
                self.last_update_time = datetime.now()
                logger.error(f"Error updating data: {e}")
            
            time.sleep(self.update_interval)
    
    def _main_loop(self, stdscr):
        """Main display loop"""
        self.screen = stdscr
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)  # Hide cursor
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Profit/Success
        curses.init_pair(2, curses.COLOR_RED, -1)    # Loss/Error
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # Warning
        curses.init_pair(4, curses.COLOR_CYAN, -1)   # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
        
        # Create windows
        self._create_windows()
        
        # Main loop
        key = 0
        while self.running and key != ord('q'):
            try:
                # Get terminal size
                self.max_y, self.max_x = self.screen.getmaxyx()
                
                # Recreate windows if terminal size changed
                self._create_windows()
                
                # Update display
                self._update_display()
                
                # Refresh screen
                self.screen.refresh()
                
                # Get user input with timeout
                self.command_window.timeout(500)  # 500ms timeout
                key = self.command_window.getch()
                
                # Handle user input
                if key == ord('r'):
                    # Refresh data immediately
                    pass
                elif key == ord('p'):
                    # Pause/resume updates
                    pass
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)
        
        # Clean up
        self.stop()
    
    def _create_windows(self):
        """Create display windows"""
        # Clear screen
        self.screen.clear()
        
        # Calculate window sizes
        height, width = self.max_y, self.max_x
        
        # Create windows
        self.metrics_window = curses.newwin(8, width, 0, 0)
        self.trade_window = curses.newwin(10, width, 8, 0)
        self.alert_window = curses.newwin(6, width // 2, 18, 0)
        self.log_window = curses.newwin(6, width // 2, 18, width // 2)
        self.status_window = curses.newwin(3, width, height - 5, 0)
        self.command_window = curses.newwin(2, width, height - 2, 0)
    
    def _update_display(self):
        """Update all display windows"""
        self._update_metrics_window()
        self._update_trade_window()
        self._update_alert_window()
        self._update_log_window()
        self._update_status_window()
        self._update_command_window()
    
    def _update_metrics_window(self):
        """Update metrics display"""
        window = self.metrics_window
        window.clear()
        
        # Get metrics
        trade_metrics = self.monitoring_dashboard.trade_metrics
        system_metrics = self.monitoring_dashboard.system_metrics
        
        # Draw header
        window.attron(curses.color_pair(5))
        window.addstr(0, 0, " TRADING BOT METRICS ".center(self.max_x))
        window.attroff(curses.color_pair(5))
        
        # Draw trade metrics
        window.addstr(1, 2, f"Total Trades: {trade_metrics.total_trades}")
        window.addstr(1, 25, f"Win Rate: ")
        if trade_metrics.win_rate >= 0.5:
            window.attron(curses.color_pair(1))
        else:
            window.attron(curses.color_pair(2))
        window.addstr(f"{trade_metrics.win_rate:.2%}")
        window.attroff(curses.A_BOLD | curses.color_pair(1) | curses.color_pair(2))
        
        window.addstr(2, 2, f"Total PnL: ")
        if trade_metrics.total_pnl >= 0:
            window.attron(curses.color_pair(1))
            window.addstr(f"${trade_metrics.total_pnl:.2f}")
        else:
            window.attron(curses.color_pair(2))
            window.addstr(f"${trade_metrics.total_pnl:.2f}")
        window.attroff(curses.color_pair(1) | curses.color_pair(2))
        
        window.addstr(2, 25, f"Daily PnL: ")
        if trade_metrics.daily_pnl >= 0:
            window.attron(curses.color_pair(1))
            window.addstr(f"${trade_metrics.daily_pnl:.2f}")
        else:
            window.attron(curses.color_pair(2))
            window.addstr(f"${trade_metrics.daily_pnl:.2f}")
        window.attroff(curses.color_pair(1) | curses.color_pair(2))
        
        window.addstr(3, 2, f"Current Balance: ${trade_metrics.current_balance:.2f}")
        window.addstr(3, 35, f"Peak Balance: ${trade_metrics.peak_balance:.2f}")
        
        window.addstr(4, 2, f"Drawdown: ")
        if trade_metrics.current_drawdown < -0.05:  # 5% drawdown
            window.attron(curses.color_pair(2))
        elif trade_metrics.current_drawdown < -0.02:  # 2% drawdown
            window.attron(curses.color_pair(3))
        else:
            window.attron(curses.color_pair(1))
        window.addstr(f"{trade_metrics.current_drawdown:.2%}")
        window.attroff(curses.color_pair(1) | curses.color_pair(2) | curses.color_pair(3))
        
        window.addstr(4, 25, f"Risk Score: ")
        if trade_metrics.risk_score > 0.7:
            window.attron(curses.color_pair(2))
        elif trade_metrics.risk_score > 0.4:
            window.attron(curses.color_pair(3))
        else:
            window.attron(curses.color_pair(1))
        window.addstr(f"{trade_metrics.risk_score:.2f}")
        window.attroff(curses.color_pair(1) | curses.color_pair(2) | curses.color_pair(3))
        
        # Draw system metrics
        window.addstr(5, 2, f"API Response: {system_metrics.api_response_time:.2f}ms")
        window.addstr(5, 35, f"Error Rate: {system_metrics.error_rate:.2%}")
        
        window.addstr(6, 2, f"Active Positions: {system_metrics.active_positions}")
        window.addstr(6, 35, f"Pending Orders: {system_metrics.pending_orders}")
        
        # Refresh window
        window.refresh()
    
    def _update_trade_window(self):
        """Update trade display"""
        window = self.trade_window
        window.clear()
        
        # Draw header
        window.attron(curses.color_pair(5))
        window.addstr(0, 0, " RECENT TRADES ".center(self.max_x))
        window.attroff(curses.color_pair(5))
        
        # Draw column headers
        window.addstr(1, 1, "TIME")
        window.addstr(1, 20, "SYMBOL")
        window.addstr(1, 30, "TYPE")
        window.addstr(1, 40, "PRICE")
        window.addstr(1, 55, "AMOUNT")
        window.addstr(1, 70, "PNL")
        
        # Draw trades
        row = 2
        for trade in self.recent_trades:
            if row >= 9:  # Limit to available rows
                break
                
            # Format timestamp
            timestamp = trade.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Get trade details
            symbol = trade.get('symbol', '')
            trade_type = trade.get('type', '')
            price = trade.get('price', 0.0)
            amount = trade.get('amount', 0.0)
            pnl = trade.get('pnl', 0.0)
            
            # Display trade
            window.addstr(row, 1, time_str)
            window.addstr(row, 20, symbol)
            
            # Color-code trade type
            if trade_type.lower() == 'buy':
                window.attron(curses.color_pair(1))
                window.addstr(row, 30, trade_type.upper())
                window.attroff(curses.color_pair(1))
            else:
                window.attron(curses.color_pair(2))
                window.addstr(row, 30, trade_type.upper())
                window.attroff(curses.color_pair(2))
            
            window.addstr(row, 40, f"${price:.4f}")
            window.addstr(row, 55, f"{amount:.6f}")
            
            # Color-code PnL
            if pnl >= 0:
                window.attron(curses.color_pair(1))
                window.addstr(row, 70, f"${pnl:.2f}")
                window.attroff(curses.color_pair(1))
            else:
                window.attron(curses.color_pair(2))
                window.addstr(row, 70, f"${pnl:.2f}")
                window.attroff(curses.color_pair(2))
            
            row += 1
        
        # Refresh window
        window.refresh()
    
    def _update_alert_window(self):
        """Update alert display"""
        window = self.alert_window
        window.clear()
        
        # Draw header
        window.attron(curses.color_pair(5))
        window.addstr(0, 0, " ALERTS ".center(self.max_x // 2))
        window.attroff(curses.color_pair(5))
        
        # Draw alerts
        row = 1
        for alert in self.recent_alerts:
            if row >= 5:  # Limit to available rows
                break
                
            # Get alert details
            severity = alert.get('severity', 'info').lower()
            message = alert.get('message', '')
            
            # Color-code by severity
            if severity == 'critical':
                window.attron(curses.color_pair(2) | curses.A_BOLD)
            elif severity == 'warning':
                window.attron(curses.color_pair(3))
            elif severity == 'info':
                window.attron(curses.color_pair(4))
            else:
                window.attron(curses.color_pair(1))
                
            # Display alert (truncate if too long)
            max_len = (self.max_x // 2) - 4
            if len(message) > max_len:
                message = message[:max_len-3] + "..."
            window.addstr(row, 1, message)
            
            # Reset attributes
            window.attroff(curses.color_pair(1) | curses.color_pair(2) | 
                          curses.color_pair(3) | curses.color_pair(4) | 
                          curses.A_BOLD)
            
            row += 1
        
        # Refresh window
        window.refresh()
    
    def _update_log_window(self):
        """Update log display"""
        window = self.log_window
        window.clear()
        
        # Draw header
        window.attron(curses.color_pair(5))
        window.addstr(0, 0, " ERROR LOG ".center(self.max_x // 2))
        window.attroff(curses.color_pair(5))
        
        # Draw errors
        row = 1
        for error in self.recent_errors:
            if row >= 5:  # Limit to available rows
                break
                
            # Get error details
            timestamp = error.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            time_str = timestamp.strftime("%H:%M:%S")
            
            message = error.get('message', '')
            
            # Display error (truncate if too long)
            max_len = (self.max_x // 2) - 10
            if len(message) > max_len:
                message = message[:max_len-3] + "..."
            
            window.attron(curses.color_pair(2))
            window.addstr(row, 1, f"{time_str} {message}")
            window.attroff(curses.color_pair(2))
            
            row += 1
        
        # Refresh window
        window.refresh()
    
    def _update_status_window(self):
        """Update status display"""
        window = self.status_window
        window.clear()
        
        # Get system status
        system_metrics = self.monitoring_dashboard.system_metrics
        network_status = system_metrics.network_status
        uptime = system_metrics.uptime
        last_update = self.last_update_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Draw status line
        window.addstr(0, 1, f"Network: ")
        if network_status.lower() == "online":
            window.attron(curses.color_pair(1))
            window.addstr(network_status)
            window.attroff(curses.color_pair(1))
        else:
            window.attron(curses.color_pair(2))
            window.addstr(network_status)
            window.attroff(curses.color_pair(2))
        
        window.addstr(0, 25, f"Uptime: {uptime:.1f} hours")
        window.addstr(0, 50, f"Last Update: {last_update}")
        
        # Draw active positions summary
        active_count = self.active_positions_count
        window.addstr(1, 1, f"Active Positions: {active_count}")
        
        # Refresh window
        window.refresh()
    
    def _update_command_window(self):
        """Update command display"""
        window = self.command_window
        window.clear()
        
        # Draw command help
        window.addstr(0, 1, "Commands: [q] Quit  [r] Refresh")
        
        # Refresh window
        window.refresh()


def start_terminal_monitor():
    """Start the terminal monitor"""
    monitor = TerminalMonitor()
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()
    except Exception as e:
        logger.error(f"Error in terminal monitor: {e}")
        monitor.stop()

if __name__ == "__main__":
    start_terminal_monitor()