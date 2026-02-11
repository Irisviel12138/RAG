"""LLM generation module with safe fallback behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | extractive
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class AnswerGenerator:
    """Generate grounded answers from retrieved evidence.

    - If provider=extractive, always returns evidence-based summary.
    - If provider=openai but OPENAI_API_KEY is missing, it falls back to extractive.
    """

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

        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _extractive_answer(
        self,
        query: str,
        evidence: List[Tuple[str, str]],
        missing_key: bool = False,
    ) -> str:
        if not evidence:
            return "未检索到有效证据，无法基于知识库回答。"

        lead = "（已使用本地 extractive 模式）" if missing_key else ""
        joined = "\n".join([f"- [{cid}] {text}" for cid, text in evidence[:3]])
        return (
            f"{lead}基于检索证据，问题“{query}”的关键信息如下：\n"
            f"{joined}\n"
            "结论：请以上述证据为准；如需更自然总结，请配置 LLM API Key。"
        )

    def _openai_answer(self, query: str, evidence: List[Tuple[str, str]]) -> str:
        from openai import OpenAI

        context = "\n".join([f"[{cid}] {text}" for cid, text in evidence])
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
