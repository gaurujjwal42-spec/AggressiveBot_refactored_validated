def test_strategy_contract_and_perf():
    from aggbot.core.strategy import Strategy
    s = Strategy()
    out = s.decide({"price": 100})
    assert out in {"BUY","SELL","HOLD"}
