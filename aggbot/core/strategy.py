from typing import Any, Dict, Optional, Callable
from .integrations import find_best_strategy_module
from .validator import validate_decide

class Strategy:
    def __init__(self, **kwargs):
        self._impl = None
        self._decide_callable: Optional[Callable] = None
        modname = find_best_strategy_module()
        if modname:
            try:
                mod = __import__(modname, fromlist=["Strategy"])
                Impl = getattr(mod, "Strategy", None)
                if Impl:
                    self._impl = Impl(**kwargs)
                    self._decide_callable = getattr(self._impl, "decide", None)
            except Exception:
                self._impl = None
        if self._decide_callable:
            try:
                validate_decide(self._decide_callable, calls=5, max_ms=25.0)
            except Exception:
                self._impl = None
                self._decide_callable = None

    def decide(self, quotes: Optional[Dict[str, Any]] = None) -> str:
        if self._decide_callable:
            try:
                if quotes is not None:
                    out = self._decide_callable(quotes)
                else:
                    out = self._decide_callable()
                if isinstance(out, str):
                    return out.upper()
                if isinstance(out, dict) and "signal" in out:
                    return str(out["signal"]).upper()
            except Exception:
                return "HOLD"
        return "HOLD"
