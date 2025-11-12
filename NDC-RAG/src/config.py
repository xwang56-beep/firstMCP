from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    data_dir: Path
    artifacts_dir: Path
    vector_store_path: Path
    embedding_model: str
    openai_api_key: str | None = None


def load_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(os.getenv("RAG_DATA_DIR", base_dir / "data")).resolve()
    artifacts_dir = Path(
        os.getenv("RAG_ARTIFACTS_DIR", base_dir / "artifacts")
    ).resolve()
    vector_store_path = Path(
        os.getenv("RAG_VECTOR_STORE_PATH", artifacts_dir / "vector_store.json")
    ).resolve()
    return AppConfig(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        vector_store_path=vector_store_path,
        embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


__all__ = ["AppConfig", "load_config"]

