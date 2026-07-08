from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION: str = "rag_documents"
    PRELOAD_DOCS_DIR: str = "data/documents"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    CONFIDENCE_THRESHOLD: float = 0.75
    TOP_K: int = Field(default=5, validation_alias=AliasChoices("DEFAULT_TOP_K", "TOP_K", "default_top_k", "top_k"))
    PROMPTS_DIR: str = "app/prompts"
    log_level: str = "INFO"

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        return self.CHUNK_OVERLAP

    @property
    def chroma_persist_dir(self) -> str:
        return self.CHROMA_PERSIST_DIR

    @property
    def chroma_collection(self) -> str:
        return self.CHROMA_COLLECTION

    @property
    def preload_docs_dir(self) -> str:
        return self.PRELOAD_DOCS_DIR


settings = Settings()

