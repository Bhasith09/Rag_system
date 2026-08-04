import json
import yaml
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer


# ---------------- LOAD DATA ----------------
def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_thresholds(path="eval/thresholds.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------- BUILD DATASET ----------------
def build_eval_dataframe():
    data = load_dataset()
    rows = []

    for item in data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        docs = hybrid_search(question, k=5)

        # ✅ FIX: RAGAS expects List[str]
        contexts = [str(d) for d in docs]

        context_text = build_context(contexts)

        answer = generate_answer(question, context_text)

        rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts
        })

    return pd.DataFrame(rows)


# ---------------- RUN EVALUATION ----------------
def run():
    df = build_eval_dataframe()

    dataset = Dataset.from_pandas(df)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall
        ]
    )

    scores = result.to_pandas().mean().to_dict()

    print("\n=== EVALUATION SCORES ===")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")

    thresholds = load_thresholds()

    failed = False
    for metric, threshold in thresholds.items():
        if scores.get(metric, 0) < threshold:
            print(f"❌ {metric} below threshold: {scores[metric]:.3f} < {threshold}")
            failed = True

    if failed:
        raise SystemExit("❌ Evaluation failed (CI BLOCKED)")

    print("✅ All metrics passed thresholds")


if __name__ == "__main__":
    run()