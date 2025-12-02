from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application configuration loaded automatically from .env.
    Access all variables via `settings.VARIABLE_NAME`.
    """

    # ==================== GOOGLE GEMINI API =====================
    GOOGLE_API_KEY: str  # required for embeddings & LLMs

    # ==================== OPENAI API (Legacy - Optional) ========
    OPENAI_API_KEY: str | None = None  # optional if transitioning from OpenAI

    # ==================== Models =========================
    EMBEDDING_MODEL: str = "models/text-embedding-004"  # Latest Google embedding model
    MODEL_NAME: str = "gemini-2.0-flash"  # Latest Gemini 2.0 model
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1000

    # ==================== Paths ==========================
    DATA_PATH: str = r"D:\ai-complaint-bot\data\complaints.csv"

    # Aliases for backward compatibility
    COMPLAINT_CSV_PATH: str = DATA_PATH
    INDEX_PATH: str = r"D:\ai-complaint-bot\embeddings\complaint_FAISS_index"
    FAISS_INDEX_PATH: str = INDEX_PATH  # backward-compatible alias

    # ==================== Chunking for embeddings =========
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # ==================== Rate Limiting ====================
    EMBEDDING_BATCH_SIZE: int = 10  # Chunks per batch
    EMBEDDING_BATCH_DELAY: float = 5.0  # Seconds between batches
    EMBEDDING_MAX_RETRIES: int = 3  # Max retry attempts on rate limit

    # ==================== Database (Optional) =============
    DB_URL: str | None = None

    # ==================== Logging / Misc ==================
    LOGGING_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Use LRU cache to avoid multiple re-instantiations
@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()