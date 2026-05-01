from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://ahody:ahody@localhost:5432/ahody"
    api_key: str = "change-me"
    voyage_api_key: str = ""
    anthropic_api_key: str = ""


settings = Settings()
