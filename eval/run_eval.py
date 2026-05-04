import os
import json
import yaml
import pandas as pd

from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Disable Chroma telemetry warnings (optional)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ----------- IMPORT YOUR PIPELINE -----------
from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer


# ----------- LOAD DATA -----------

def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_thresholds(path="eval/thresholds.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------- BUILD EVAL DATA -----------

def build_eval_dataframe():
    data = load_dataset()
    rows = []

    for item in data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Step 1: Retrieve documents
        docs = hybrid_search(question, k=5)

        # Step 2: FIX — must be List[str]
        contexts = [str(d) for d in docs] if docs else []

        # Step 3: Build context string for LLM
        context_text = build_context(contexts)

        # Step 4: Generate answer
        answer = generate_answer(question, context_text)

        # Step 5: Store result
        rows.append({
            "question": str(question),
            "ground_truth": str(ground_truth),
            "answer": str(answer),
            "contexts": contexts
        })

    return pd.DataFrame(rows)


# ----------- RUN EVALUATION -----------

def run():
    df = build_eval_dataframe()

    # FIX: Explicit schema for RAGAS
    features = Features({
        "question": Value("string"),
        "ground_truth": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string"))
    })

    dataset = Dataset.from_pandas(df, features=features)

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall
        ]
    )

    # Compute mean scores
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    print("\n=== EVALUATION SCORES ===")
    for metric, value in scores.items():
        print(f"{metric}: {value:.3f}")

    # ----------- THRESHOLD CHECK -----------

    thresholds = load_thresholds()
    failed = False

    for metric, threshold in thresholds.items():
        if metric in scores and scores[metric] < threshold:
            print(f"❌ {metric} below threshold: {scores[metric]:.3f} < {threshold}")
            failed = True

    if failed:
        raise SystemExit("❌ Build failed due to low evaluation scores")

    print("✅ All evaluation scores are above thresholds")


# ----------- MAIN -----------

if __name__ == "__main__":
    run()