from __future__ import annotations
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "trade-ml-service"
    app_version: str = "0.1.0"
    app_port: int = Field(default=3099, alias="APP_PORT")

    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_name: str = Field(default="tradding_db", alias="DB_NAME")
    db_username: str = Field(default="root", alias="DB_USERNAME")
    db_password: str = Field(default="ellenverma", alias="DB_PASSWORD")
    database_url_override: Optional[str] = Field(default=None, alias="DATABASE_URL")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    default_prediction_horizon_bars: int = Field(default=15, alias="DEFAULT_PREDICTION_HORIZON_BARS")
    default_buy_threshold_pct: float = Field(default=0.5, alias="DEFAULT_BUY_THRESHOLD_PCT")
    default_sell_threshold_pct: float = Field(default=-0.5, alias="DEFAULT_SELL_THRESHOLD_PCT")

    model_directory: str = Field(default="./models", alias="MODEL_DIRECTORY")

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override

        return (
            f"mysql+pymysql://{self.db_username}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )


settings = Settings()
