import time
import typer
from rich.console import Console
from .settings import Settings
from .core.engine import Engine

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def run(dry_run: bool = typer.Option(False), profile: bool = typer.Option(False), cycles: int = 1):
    cfg = Settings()
    console.log(f"[bold green]AggBot booting[/] env={cfg.runtime_env} dry_run={dry_run}")
    engine = Engine(cfg, dry_run=dry_run)

    if profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    for i in range(cycles):
        console.log(f"Cycle {i+1}/{cycles}")
        engine.tick()
        time.sleep(cfg.loop_sleep_s)

    if profile:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
        console.rule("Top 30 hotspots")
        console.print(s.getvalue())

@app.command()
def doctor():
    cfg = Settings()
    console.print("Config loaded. env=%s log=%s" % (cfg.runtime_env, cfg.log_level))

if __name__ == "__main__":
    app()