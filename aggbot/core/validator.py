import time
from typing import Any, Dict, Optional, Callable

ALLOWED = {"BUY","SELL","HOLD"}

class ValidationError(Exception): ...
class PerformanceError(Exception): ...

def validate_decide(decide: Callable[..., str], *, calls: int = 10, max_ms: float = 20.0) -> None:
    """Validate output domain and latency of decide().
    - calls: repeat calls to catch flakiness
    - max_ms: per-call budget in milliseconds
    """
    quotes = {"price": 100.0, "bid": 99.9, "ask": 100.1, "ts": 0}
    for i in range(calls):
        t0 = time.perf_counter()
        out = None
        try:
            out = decide(quotes)
        except TypeError:
            out = decide()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms > max_ms:
            raise PerformanceError(f"decide() too slow: {dt_ms:.2f}ms > {max_ms}ms (iter {i})")
        if not isinstance(out, str) or out.upper() not in ALLOWED:
            raise ValidationError(f"decide() must return one of {ALLOWED}, got {out!r}")
