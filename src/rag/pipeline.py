"""End-to-end RAG pipeline skeleton."""

from dataclasses import dataclass, field
from typing import Dict, List

from .chunking import ChunkConfig, chunk_text
from .embeddings import EmbeddingConfig, EmbeddingProvider
from .index import InMemoryIndex, IndexConfig
from .rerank import rerank


@dataclass
class PipelineConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)


class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedder = EmbeddingProvider(config.embedding)
        self.index = InMemoryIndex(config.index)
        self._chunks: Dict[str, str] = {}

    def ingest(self, doc_id: str, content: str) -> None:
        chunks = chunk_text(content, self.config.chunk)
        vectors = self.embedder.embed(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            cid = f"{doc_id}:{i}"
            self._chunks[cid] = chunk
            self.index.upsert(cid, vector)

    def answer(self, query: str, top_k: int = 8, top_n: int = 3) -> Dict[str, List[str]]:
        qv = self.embedder.embed([query])[0]
        candidate_ids = self.index.search(qv, top_k=top_k)
        passages = [self._chunks[cid] for cid in candidate_ids if cid in self._chunks]
        ranked = rerank(query, passages, top_n=top_n)
        return {
            "query": query,
            "evidence": [p for p, _ in ranked],
            "answer": "[stub] 基于证据生成答案（待接入真实 LLM）",
        }
