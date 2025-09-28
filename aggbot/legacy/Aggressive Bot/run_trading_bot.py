import sys
import os
import traceback
from datetime import datetime

# Import the centralized, production-ready logger
from enhanced_logger import EnhancedLogger

# Initialize the logger at the application's entry point.
# This single instance should be passed to or imported by all other modules.
logger = EnhancedLogger()

# Use the new logger for system events
logger.log_system_event("BOT_STARTUP", {"message": "Starting trading bot in standalone mode"})
print(f"INFO: Logs are being saved to the '{logger.log_dir}' directory.")

# Import configuration and check API credentials
import config
logger.log_info("Loaded configuration")

# Check if API credentials are set
if not config.API_KEY or not config.API_SECRET:
    logger.log_error(
        "API credentials are not set. Trading functionality will be unavailable.",
        "CONFIG_ERROR"
    )
    logger.log_info(
        "Please set API_KEY and API_SECRET in a .env file for local development, "
        "or as environment variables in production."
    )
    sys.exit(1)

logger.log_info(f"Using {'TESTNET' if config.USE_TESTNET else 'LIVE'} Binance API")

if not getattr(config, 'USE_TESTNET', True):
    logger.log_error("LIVE MODE - Real trades will be executed!", "OPERATIONAL_MODE")
else:
    logger.log_info("TEST MODE - No real trades will be executed")

# Import and run the main bot function
try:
    import main_bot
    logger.log_info("Starting main trading bot loop...")
    main_bot.run_bot()
except KeyboardInterrupt:
    logger.log_system_event("BOT_SHUTDOWN", {"message": "Bot stopped by user."})
except Exception as e:
    logger.log_error(
        f"Fatal error in trading bot: {e}",
        "FATAL_ERROR",
        {"traceback": traceback.format_exc()}
    )
    sys.exit(1)