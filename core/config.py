from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Gepato-AI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False          # True — SQL so'rovlari logga (faqat lokal)

    # Web: bir xil domen (frontend shu API'dan) — CORS kerak emas. Boshqa domen kerak bo'lsa:
    # ALLOWED_ORIGINS=https://a.uz,https://b.uz
    ALLOWED_ORIGINS: str = ""
    ENABLE_DOCS: bool = False    # /docs, /openapi.json — productionda standart yopiq

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://gepato:gepato@localhost:5432/gepato"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379"

    # Auth
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12  # 12 soat
    # Birinchi admin (deploy/install.sh tasodifiy parol bilan yozadi). Ishga tushganda yo'q bo'lsa yaratiladi;
    # mavjud hisobning paroli o'zgartirilmaydi.
    BOOTSTRAP_ADMIN_EMAIL: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = ""

    # AI
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-5"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"  # 2.5-flash yangi kalitlarga yopilgan (2026-11)
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
        extra = "ignore"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ("production", "prod")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    def production_problems(self) -> list[str]:
        """Productionda ishga tushirishni TO'XTATADIGAN sozlamalar (xato sozlangan server ochiq qolmasin)."""
        if not self.is_production:
            return []
        bad = []
        if self.SECRET_KEY in WEAK_SECRETS or len(self.SECRET_KEY) < 32:
            bad.append("SECRET_KEY zaif yoki standart (kamida 32 belgi, tasodifiy)")
        if "*" in self.cors_origins:
            bad.append("ALLOWED_ORIGINS='*' — productionda ruxsat yo'q")
        if "gepato:gepato@" in self.DATABASE_URL:
            bad.append("DATABASE_URL standart parol bilan (gepato:gepato)")
        if self.BOOTSTRAP_ADMIN_PASSWORD in ("change-me-admin",):
            bad.append("BOOTSTRAP_ADMIN_PASSWORD namunaviy qiymatda")
        return bad


WEAK_SECRETS = {"", "change-me-in-production", "change-this-to-random-64-char-string", "secret"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
