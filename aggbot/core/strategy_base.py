from typing import Protocol, Optional, Any, Dict

class StrategyBase(Protocol):
    def decide(self, quotes: Optional[Dict[str, Any]] = None) -> str: ...
