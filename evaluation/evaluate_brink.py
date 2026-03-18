#!/usr/bin/env python3
"""
Official-style BRINK evaluation script.

This evaluator expects:
1. A gold JSON file with entries like:
   {
     "id": "q1",
     "question": "...",
     "answers": ["Paris", "London"],
     "hard_answer": "Paris"
   }

2. A prediction JSON file with entries like:
   {
     "id": "q1",
     "raw_output": "Paris, London"
   }

The evaluator:
- accepts the model's raw output string
- splits it into candidate answers
- normalizes predictions and gold answers
- computes:
    Hits@Any
    Precision
    Recall
    F1
    Hits@Hard
    HHR
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """
    Normalize an answer string following the paper description:
    - lowercase
    - remove articles (a, an, the)
    - remove punctuation
    - remove <pad>
    - collapse extra whitespace
    """
    text = text.lower()
    text = text.replace("<pad>", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in ARTICLES]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_raw_output(
    raw_output: str,
    split_on_spaces: bool = False,
) -> List[str]:
    """
    Convert a raw output string into candidate answer strings.

    Default behavior:
    - split on commas and newlines

    Optional behavior:
    - also split on whitespace (can be useful for some datasets,
      but may break multi-word answers such as 'New York')
    """
    if raw_output is None:
        return []

    raw_output = raw_output.strip()
    if not raw_output:
        return []

    if split_on_spaces:
        parts = re.split(r"[,\n\r\t ]+", raw_output)
    else:
        parts = re.split(r"[,\n\r]+", raw_output)

    parts = [p.strip() for p in parts if p.strip()]
    return parts


def deduplicate_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def process_prediction(
    raw_output: str,
    split_on_spaces: bool = False,
) -> List[str]:
    candidates = split_raw_output(raw_output, split_on_spaces=split_on_spaces)
    normalized = [normalize_answer(x) for x in candidates]
    normalized = [x for x in normalized if x]
    return deduplicate_keep_order(normalized)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold(path: str | Path) -> Dict[Any, Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Gold file must be a JSON list.")

    gold: Dict[Any, Dict[str, Any]] = {}
    for ex in data:
        if "id" not in ex:
            raise ValueError(f"Gold example missing 'id': {ex}")
        if "answers" not in ex:
            raise ValueError(f"Gold example missing 'answers': {ex}")
        if "hard_answer" not in ex:
            raise ValueError(f"Gold example missing 'hard_answer': {ex}")

        qid = ex["id"]
        answers = ex["answers"]
        hard_answer = ex["hard_answer"]

        if not isinstance(answers, list):
            raise ValueError(f"'answers' must be a list for id={qid}")

        norm_answers = {
            normalize_answer(str(a))
            for a in answers
            if normalize_answer(str(a))
        }
        norm_hard = normalize_answer(str(hard_answer))

        gold[qid] = {
            "answers": norm_answers,
            "hard_answer": norm_hard,
        }

    return gold


def load_predictions(
    path: str | Path,
    split_on_spaces: bool = False,
) -> Dict[Any, List[str]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Prediction file must be a JSON list.")

    preds: Dict[Any, List[str]] = {}
    for ex in data:
        if "id" not in ex:
            raise ValueError(f"Prediction example missing 'id': {ex}")
        if "raw_output" not in ex:
            raise ValueError(f"Prediction example missing 'raw_output': {ex}")

        qid = ex["id"]
        raw_output = str(ex["raw_output"])
        preds[qid] = process_prediction(
            raw_output,
            split_on_spaces=split_on_spaces,
        )

    return preds


def compute_example_metrics(
    pred_set: Set[str],
    gold_set: Set[str],
    hard_answer: str,
) -> Dict[str, float]:
    overlap = pred_set & gold_set

    hits_any = 1.0 if overlap else 0.0
    hits_hard = 1.0 if hard_answer in pred_set else 0.0

    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    recall = len(overlap) / len(gold_set) if gold_set else 0.0
    f1 = (2.0 * len(overlap) / (len(pred_set) + len(gold_set))) if (len(pred_set) + len(gold_set)) > 0 else 0.0

    return {
        "Hits@Any": hits_any,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Hits@Hard": hits_hard,
    }


def evaluate(
    gold: Dict[Any, Dict[str, Any]],
    preds: Dict[Any, List[str]],
) -> Dict[str, float]:
    if not gold:
        raise ValueError("Gold file is empty.")

    total_hits_any = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_hits_hard = 0.0
    num_missing_predictions = 0

    for qid, gold_ex in gold.items():
        pred_list = preds.get(qid, [])
        if qid not in preds:
            num_missing_predictions += 1

        pred_set = set(pred_list)
        gold_set = gold_ex["answers"]
        hard_answer = gold_ex["hard_answer"]

        metrics = compute_example_metrics(pred_set, gold_set, hard_answer)

        total_hits_any += metrics["Hits@Any"]
        total_precision += metrics["Precision"]
        total_recall += metrics["Recall"]
        total_f1 += metrics["F1"]
        total_hits_hard += metrics["Hits@Hard"]

    n = len(gold)

    results = {
        "num_examples": n,
        "num_missing_predictions": num_missing_predictions,
        "Hits@Any": total_hits_any / n,
        "Precision": total_precision / n,
        "Recall": total_recall / n,
        "F1": total_f1 / n,
        "Hits@Hard": total_hits_hard / n,
    }

    results["HHR"] = (
        results["Hits@Hard"] / results["Hits@Any"]
        if results["Hits@Any"] > 0
        else 0.0
    )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BRINK predictions.")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold JSON file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction JSON file.")
    parser.add_argument(
        "--split-on-spaces",
        action="store_true",
        help="Also split raw outputs on spaces. Use with caution for multi-word answers.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gold = load_gold(args.gold)
    preds = load_predictions(args.pred, split_on_spaces=args.split_on_spaces)
    results = evaluate(gold, preds)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
