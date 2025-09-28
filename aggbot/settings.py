from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    runtime_env: str = Field(default="local", description="local|staging|prod")
    log_level: str = "INFO"
    loop_sleep_s: float = 1.0

    # Example API creds (override in .env)
    API_KEY: str | None = None
    API_SECRET: str | None = None
    BASE_URL: str | None = None

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }