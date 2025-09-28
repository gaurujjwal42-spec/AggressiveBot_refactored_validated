import logging
from ..settings import Settings
from .strategy import Strategy
from .broker import Broker

log = logging.getLogger("aggbot.engine")

class Engine:
    def __init__(self, cfg: Settings, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.strategy = Strategy()
        self.broker = Broker()

    def tick(self):
        quotes = None
        try:
            quotes = self.broker.fetch_quotes()
        except Exception:
            quotes = None
        signal = self.strategy.decide(quotes)
        log.info(f"Signal: {signal}")
        if signal in ("BUY","SELL"):
            if self.dry_run:
                log.info(f"[DRY RUN] Would {signal}")
            else:
                res = self.broker.place_order(side=signal)
                log.info(f"Order result: {res}")
