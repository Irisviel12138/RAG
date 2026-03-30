"""Top-k ablation experiment for RAG on TriviaQA (unfiltered).

Implements the assignment setup shown in the screenshots:
- Dataset: mandarjoshi/trivia_qa (unfiltered)
- Retriever: BM25
- Generator: google/flan-t5-base
- Metrics: EM/F1, Hit@k, faithfulness categories
- Plots:
  1) Accuracy(EM) vs Top-k
  2) Supported Rate vs Top-k
  3) Correct but Unsupported vs Top-k
"""

from __future__ import annotations

import argparse
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_TOPK_VALUES = [1, 3, 5, 10, 20]


@dataclass
class EvalSample:
    """One QA sample with candidate retrieval passages."""

    question: str
    answers: List[str]
    passages: List[str]


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization for EM/F1."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in golds if g))


def f1_score(prediction: str, golds: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        common_count = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
        precision = common_count / len(pred_tokens)
        recall = common_count / len(gold_tokens)
        if precision + recall == 0:
            continue
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_any_answer(text: str, answers: Sequence[str]) -> bool:
    norm_text = normalize_answer(text)
    return any(ans and normalize_answer(ans) in norm_text for ans in answers)


def build_eval_samples(raw_dataset: Dataset, num_samples: int) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for row in raw_dataset:
        question = row.get("question", "")
        answer_info = row.get("answer", {}) or {}
        aliases = answer_info.get("aliases", []) or []
        if not question or not aliases:
            continue

        search_results = row.get("search_results", {}) or {}
        descriptions = search_results.get("description", []) or []
        snippets = search_results.get("search_context", []) or []
        passages = [p for p in descriptions + snippets if isinstance(p, str) and p.strip()]

        if not passages:
            continue

        samples.append(EvalSample(question=question, answers=list(dict.fromkeys(aliases)), passages=passages))
        if len(samples) >= num_samples:
            break
    return samples


def tokenize_for_bm25(text: str) -> List[str]:
    text = normalize_answer(text)
    return text.split() if text else []


def retrieve_topk_bm25(question: str, passages: Sequence[str], k: int) -> List[str]:
    corpus_tokens = [tokenize_for_bm25(p) for p in passages]
    if not any(corpus_tokens):
        return list(passages[:k])
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize_for_bm25(question))
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:k]]


def generate_answer(
    question: str,
    evidence: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    context = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(evidence))
    prompt = (
        "Answer the question based only on the evidence. "
        "If evidence is insufficient, output 'I don't know'.\n"
        f"Question: {question}\n"
        f"Evidence:\n{context}\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def classify_faithfulness(is_em: bool, supported: bool) -> str:
    if is_em and supported:
        return "correct_supported"
    if is_em and not supported:
        return "correct_unsupported"
    return "incorrect"


def evaluate_for_k(
    samples: Sequence[EvalSample],
    k: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
) -> Dict[str, float]:
    em_total = 0.0
    f1_total = 0.0
    hit_total = 0.0
    cs = 0
    cu = 0
    incorrect = 0

    for sample in tqdm(samples, desc=f"Evaluating top-k={k}", leave=False):
        retrieved = retrieve_topk_bm25(sample.question, sample.passages, k=k)
        joined_retrieved = "\n".join(retrieved)
        hit_k = float(contains_any_answer(joined_retrieved, sample.answers))

        pred = generate_answer(sample.question, retrieved, tokenizer, model, device)
        em = exact_match(pred, sample.answers)
        f1 = f1_score(pred, sample.answers)
        supported = contains_any_answer(joined_retrieved, [pred])
        faith_class = classify_faithfulness(bool(em), supported)

        em_total += em
        f1_total += f1
        hit_total += hit_k
        if faith_class == "correct_supported":
            cs += 1
        elif faith_class == "correct_unsupported":
            cu += 1
        else:
            incorrect += 1

    n = max(1, len(samples))
    return {
        "k": k,
        "em": em_total / n,
        "f1": f1_total / n,
        "hit_at_k": hit_total / n,
        "supported_rate": cs / n,
        "correct_unsupported_rate": cu / n,
        "incorrect_rate": incorrect / n,
    }


def plot_curves(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []

    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["em"], marker="o")
    plt.title("Accuracy (EM) vs Top-k")
    plt.xlabel("Top-k")
    plt.ylabel("EM")
    plt.grid(alpha=0.3)
    p1 = out_dir / "accuracy_vs_topk.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(p1)

    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["supported_rate"], marker="o", color="green")
    plt.title("Supported Rate vs Top-k")
    plt.xlabel("Top-k")
    plt.ylabel("Correct & Supported Rate")
    plt.grid(alpha=0.3)
    p2 = out_dir / "supported_vs_topk.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(p2)

    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["correct_unsupported_rate"], marker="o", color="orange")
    plt.title("Correct but Unsupported vs Top-k")
    plt.xlabel("Top-k")
    plt.ylabel("Correct but Unsupported Rate")
    plt.grid(alpha=0.3)
    p3 = out_dir / "correct_unsupported_vs_topk.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(p3)

    return plot_paths


def run_experiment(
    split: str,
    num_samples: int,
    topk_values: Sequence[int],
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=split)
    samples = build_eval_samples(dataset, num_samples=num_samples)
    if not samples:
        raise RuntimeError("No valid samples were found in TriviaQA unfiltered split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    for k in topk_values:
        rows.append(evaluate_for_k(samples, k, tokenizer, model, device))

    df = pd.DataFrame(rows).sort_values("k")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "topk_results.csv", index=False)
    plot_curves(df, output_dir)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k ablation for TriviaQA RAG.")
    parser.add_argument("--split", default="validation", help="TriviaQA split: train/validation")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of QA samples to evaluate")
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=DEFAULT_TOPK_VALUES,
        help="Top-k values to test, e.g. --topk 1 3 5 10 20",
    )
    parser.add_argument("--model", default="google/flan-t5-base", help="Generator model name")
    parser.add_argument("--output-dir", default="artifacts/topk_ablation", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_experiment(
        split=args.split,
        num_samples=args.num_samples,
        topk_values=args.topk,
        model_name=args.model,
        output_dir=Path(args.output_dir),
    )
    print("\n=== Top-k Ablation Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
