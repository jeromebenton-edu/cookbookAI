from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    frontend_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ],
        env="FRONTEND_ORIGINS",
    )
    model_dir: Optional[str] = Field(default=None, env="MODEL_DIR")
    dataset_dir: Optional[str] = Field(default=None, env="DATASET_DIR")
    warmup: bool = Field(default=True, env="WARMUP")
    prewarm: bool = Field(default=False, env="COOKBOOKAI_PREWARM")
    cache_dir: str = Field(default="backend/cache", env="CACHE_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def parse_origins(cls, raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    raw = os.environ.get("FRONTEND_ORIGINS")
    settings = Settings()
    if raw:
        settings.frontend_origins = Settings.parse_origins(raw)
    return settings
