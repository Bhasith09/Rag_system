import json
import yaml
import pandas as pd
import os

from datasets import Dataset
from ragas import evaluate

# RAGAS metrics (new correct import)
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)

# your pipeline
from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer


# =========================
# GROQ LLM FOR RAGAS EVAL
# =========================
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper


groq_eval_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

ragas_llm = LangchainLLMWrapper(groq_eval_llm)


# =========================
# LOAD DATA
# =========================

def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_thresholds(path="eval/thresholds.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# BUILD EVAL DATASET
# =========================

def build_eval_dataframe():
    data = load_dataset()
    rows = []

    for item in data:
        question = item["question"]
        ground_truth = item["answer"]

        docs = hybrid_search(question, k=5)

        # safe fallback
        contexts = docs if docs else ["no context found"]

        # convert context list to string for LLM
        context_text = build_context(contexts)

        # generate answer using your RAG pipeline
        answer = generate_answer(question, context_text)

        rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts
        })

    return pd.DataFrame(rows)


# =========================
# RUN EVALUATION
# =========================

def run():
    df = build_eval_dataframe()
    dataset = Dataset.from_pandas(df)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall
        ],
        llm=ragas_llm
    )

    scores = result.to_pandas().mean().to_dict()

    print("\n=== Evaluation Scores ===")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")

    # =========================
    # THRESHOLD CHECK
    # =========================
    thresholds = load_thresholds()
    failed = False

    for metric, threshold in thresholds.items():
        if metric in scores and scores[metric] < threshold:
            print(f"❌ {metric} below threshold: {scores[metric]:.3f} < {threshold}")
            failed = True

    if failed:
        raise SystemExit("Build failed due to low evaluation scores")

    print("✅ All evaluation scores are above thresholds")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run()