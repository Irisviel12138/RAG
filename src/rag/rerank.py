"""Cross-encoder reranking placeholders."""

from typing import List, Tuple


def rerank(query: str, passages: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """Return naively scored passages as a baseline stub."""
    scored = [(p, float(len(set(query) & set(p)))) for p in passages]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
