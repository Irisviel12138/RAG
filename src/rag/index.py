"""Index abstraction for FAISS/Chroma experimentation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class IndexConfig:
    backend: str = "faiss"


class InMemoryIndex:
    def __init__(self, config: IndexConfig):
        if config.backend not in {"faiss", "chroma"}:
            raise ValueError(f"Unsupported index backend: {config.backend}")
        self.config = config
        self._store: Dict[str, List[float]] = {}

    def upsert(self, doc_id: str, vector: List[float]) -> None:
        self._store[doc_id] = vector

    def search(self, _query_vector: List[float], top_k: int = 5) -> List[str]:
        return list(self._store.keys())[:top_k]
