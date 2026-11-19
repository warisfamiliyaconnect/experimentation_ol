from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    olmo_model_path: str = "artifacts/OLMo-2-0425-1B-FP16"
    olmo_config_path: str = "configs/inference_fp16.yaml"
    olmo_device: Optional[str] = None
    olmo_server_url: Optional[str] = None
    olmo_log_file: str = "data/processed/inference_response.json"

settings = Settings()
