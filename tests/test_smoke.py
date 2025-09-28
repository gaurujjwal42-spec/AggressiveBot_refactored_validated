from aggbot.core.engine import Engine
from aggbot.settings import Settings

def test_engine_smoke():
    eng = Engine(Settings(), dry_run=True)
    eng.tick()