"""Embedding provider abstraction for OpenAI/BGE comparison."""

from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingConfig:
    provider: str = "bge"
    model: str = "BAAI/bge-large-zh-v1.5"


class EmbeddingProvider:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Placeholder embedding method.

        Replace with real API/model calls in production.
        """
        if self.config.provider not in {"openai", "bge"}:
            raise ValueError(f"Unsupported embedding provider: {self.config.provider}")
        return [[float(len(t) % 10)] * 8 for t in texts]
