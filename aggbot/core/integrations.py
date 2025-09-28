import importlib, pkgutil, inspect, re
from typing import Optional

def _iter_modules(package_name: str):
    pkg = importlib.import_module(package_name)
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        yield m.name

def _extract_score(obj) -> float:
    score = getattr(obj, "SCORE", None)
    if isinstance(score, (int, float)):
        return float(score)
    doc = (inspect.getdoc(obj) or "")
    m = re.search(r"SCORE\s*:\s*([-+]?[0-9]*\.?[0-9]+)", doc, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0

def find_best_strategy_module() -> Optional[str]:
    best = (0.0, None)
    try:
        for name in _iter_modules("aggbot.legacy"):
            try:
                mod = importlib.import_module(name)
                Impl = getattr(mod, "Strategy", None)
                if Impl and hasattr(Impl, "decide"):
                    sc = max(_extract_score(mod), _extract_score(Impl))
                    if sc >= best[0]:
                        best = (sc, name)
            except Exception:
                continue
    except Exception:
        return None
    return best[1]

def find_broker_module() -> Optional[str]:
    for name in _iter_modules("aggbot.legacy"):
        try:
            mod = importlib.import_module(name)
            cls = getattr(mod, "Broker", None)
            if cls and hasattr(cls, "place_order"):
                return name
        except Exception:
            continue
    return None
