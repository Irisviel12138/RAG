"""Chunking strategies for engineering-grade RAG experiments."""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ChunkConfig:
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 100


def chunk_text(text: str, config: ChunkConfig) -> List[str]:
    """Minimal baseline chunking utility.

    This baseline intentionally keeps implementation simple so that you can
    replace it with LangChain/LlamaIndex splitters later.
    """
    if not text.strip():
        return []

    if config.strategy not in {"fixed", "recursive", "semantic"}:
        raise ValueError(f"Unsupported strategy: {config.strategy}")

    step = max(1, config.chunk_size - config.chunk_overlap)
    return [text[i : i + config.chunk_size] for i in range(0, len(text), step)]


def batch_chunk(texts: Iterable[str], config: ChunkConfig) -> List[str]:
    output: List[str] = []
    for text in texts:
        output.extend(chunk_text(text, config))
    return output
