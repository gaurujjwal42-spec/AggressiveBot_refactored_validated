import threading
import logging
from waitress import serve

import main_bot
from web_server import app

def run_web_server():
    """Runs the Flask web server using waitress."""
    logging.info("Starting web server on http://127.0.0.1:5001")
    serve(app, host='127.0.0.1', port=5001)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a thread for the trading bot
    bot_thread = threading.Thread(target=main_bot.run_bot, name="TradingBot")

    # Create a thread for the web server
    server_thread = threading.Thread(target=run_web_server, name="WebServer")
    server_thread.daemon = True # Allows main thread to exit even if server is running

    bot_thread.start()
    server_thread.start()
