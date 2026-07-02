from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool

    GROQ_API_KEY: str = ""

    VECTOR_DB_PATH: str = "./vector_db"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    DATA_PATH: str = "./data"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()