"""LLM generation module with safe fallback behavior."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple
from urllib import error, request
import json


@dataclass
class LLMConfig:
    provider: str = "extractive"  # extractive | openai | ollama
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"


class AnswerGenerator:
    """Generate grounded answers from retrieved evidence."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, query: str, evidence: List[Tuple[str, str]]) -> str:
        if self.config.provider == "extractive":
            return self._extractive_answer(query, evidence)

        if self.config.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                return self._extractive_answer(query, evidence, missing_key=True)
            return self._openai_answer(query, evidence)

        if self.config.provider == "ollama":
            return self._ollama_answer(query, evidence)

        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _extractive_answer(
        self,
        query: str,
        evidence: List[Tuple[str, str]],
        missing_key: bool = False,
    ) -> str:
        if not evidence:
            return "未检索到有效证据，无法基于知识库回答。"

        top = self._select_key_evidence(query, evidence)
        if not top:
            top = evidence[:2]

        bullets = "\n".join([f"- {self._truncate(self._clean_inline(text), 180)} [{cid}]" for cid, text in top])
        prefix = "（未检测到 API Key，使用无API extractive回答）\n" if missing_key else ""

        return (
            f"{prefix}基于证据，可得：\n"
            f"{bullets}\n"
            "结论：以上为检索证据支持的答案；若需更流畅总结可切换 OpenAI/Ollama。"
        )

    def _select_key_evidence(self, query: str, evidence: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        query_terms = {t for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", query.lower()) if len(t) > 1}
        scored: List[Tuple[float, Tuple[str, str]]] = []
        for cid, text in evidence:
            clean = self._clean_inline(text).lower()
            terms = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", clean))
            overlap = len(query_terms & terms)
            density = overlap / max(1, len(terms))
            score = overlap * 2 + density
            scored.append((score, (cid, text)))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [item for score, item in scored if score > 0]
        return best[:2]

    def _openai_answer(self, query: str, evidence: List[Tuple[str, str]]) -> str:
        from openai import OpenAI

        context = "\n".join([f"[{cid}] {self._clean_inline(text)}" for cid, text in evidence])
        system_prompt = (
            "你是一个严谨的RAG回答助手。"
            "必须仅基于给定证据回答；若证据不足，请明确说不知道。"
            "回答末尾必须保留引用编号，如[doc-1:0]。"
        )
        user_prompt = (
            f"问题：{query}\n\n"
            f"证据：\n{context}\n\n"
            "请输出简洁中文答案，并给出引用。"
        )

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or "LLM 未返回内容。"

    def _ollama_answer(self, query: str, evidence: List[Tuple[str, str]]) -> str:
        if not evidence:
            return "未检索到有效证据，无法基于知识库回答。"

        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "仅根据证据回答，不要编造；回答末尾带引用编号。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n证据：\n"
                    + "\n".join([f"[{cid}] {self._clean_inline(text)}" for cid, text in evidence]),
                },
            ],
            "options": {"temperature": self.config.temperature},
        }

        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "").strip()
                return content or "Ollama 未返回内容。"
        except error.URLError:
            return "无法连接到本地 Ollama（http://localhost:11434）。请先启动 Ollama，或切换到 extractive 模式。"

    @staticmethod
    def _clean_inline(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("�", "")
        return text

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
