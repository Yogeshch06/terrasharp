import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/model.onnx")
    HF_MODEL_REPO: str = os.getenv("HF_MODEL_REPO", "Yogeshch2006/terrasharp")
    HF_MODEL_FILENAME: str = os.getenv("HF_MODEL_FILENAME", "terrasharp_swinir_sentinel2.onnx")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)

    COPERNICUS_CLIENT_ID: Optional[str] = os.getenv("COPERNICUS_CLIENT_ID", None)
    COPERNICUS_CLIENT_SECRET: Optional[str] = os.getenv("COPERNICUS_CLIENT_SECRET", None)

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", 50))
    TILE_SIZE: int = int(os.getenv("TILE_SIZE", 256))
    UPSCALE_FACTOR: int = int(os.getenv("UPSCALE_FACTOR", 4))
    MC_DROPOUT_PASSES: int = int(os.getenv("MC_DROPOUT_PASSES", 2))

    ALLOWED_ORIGIN: str = os.getenv("ALLOWED_ORIGIN", "*")
    DB_PATH: str = os.getenv("DB_PATH", "./data/jobs.db")
    TMP_DIR: str = os.getenv("TMP_DIR", "./tmp")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
