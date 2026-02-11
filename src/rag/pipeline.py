"""End-to-end RAG pipeline skeleton with pluggable LLM generation."""

from dataclasses import dataclass, field
from typing import Dict, List

from .chunking import ChunkConfig, chunk_text
from .embeddings import EmbeddingConfig, EmbeddingProvider
from .index import InMemoryIndex, IndexConfig
from .llm import AnswerGenerator, LLMConfig
from .rerank import rerank


@dataclass
class PipelineConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedder = EmbeddingProvider(config.embedding)
        self.index = InMemoryIndex(config.index)
        self.generator = AnswerGenerator(config.llm)
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
        passages = [(cid, self._chunks[cid]) for cid in candidate_ids if cid in self._chunks]
        ranked = rerank(query, [text for _, text in passages], top_n=top_n)

        text_to_cid = {text: cid for cid, text in passages}
        evidence_pairs = [(text_to_cid.get(text, "unknown"), text) for text, _ in ranked]
        answer = self.generator.generate(query, evidence_pairs)

        return {
            "query": query,
            "evidence": [f"[{cid}] {text}" for cid, text in evidence_pairs],
            "answer": answer,
        }
