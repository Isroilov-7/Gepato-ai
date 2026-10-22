from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Gepato-AI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://gepato:gepato@localhost:5432/gepato"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379"

    # Auth
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 soat

    # AI
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-5"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    AI_PROVIDER: str = "auto"  # auto | gemini | claude

    # Audio
    UPLOAD_DIR: str = "uploads"
    MAX_AUDIO_SIZE_MB: int = 10
    MIN_AUDIO_DURATION_SEC: float = 5.0
    MAX_AUDIO_DURATION_SEC: float = 60.0
    MIN_SNR_DB: float = 10.0  # shovqin filtri

    # ML
    MODEL_DIR: str = "models"
    MODEL_RETRAIN_THRESHOLD: int = 50  # yangi labeled sample soni
    MIN_SAMPLES_FOR_TRAINING: int = 20

    # Active learning
    UNCERTAINTY_THRESHOLD: float = 0.3  # [0.35, 0.65] → shifokordan label so'ra

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
