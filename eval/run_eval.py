import json
import yaml
import pandas as pd
import os

from datasets import Dataset
from ragas import evaluate

# Metrics
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)

# Your pipeline
from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer

# LangChain Document
from langchain_core.documents import Document


# =========================
# GROQ LLM (FOR RAGAS JUDGING)
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
# LOCAL EMBEDDINGS
# =========================

from ragas.embeddings import HuggingfaceEmbeddings

embeddings = HuggingfaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


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
# BUILD EVALUATION DATASET
# =========================

def build_eval_dataframe():

    data = load_dataset()

    rows = []

    for item in data:

        question = item["question"]
        ground_truth = item["answer"]

        docs = hybrid_search(question, k=5)

        # Fallback if nothing is retrieved
        if not docs:
            docs = [
                Document(
                    page_content="No relevant context found.",
                    metadata={
                        "source": "None",
                        "page": 0,
                        "paragraph": 0
                    }
                )
            ]

        context = build_context(docs)

        answer = generate_answer(question, context)

        rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs]
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
        llm=ragas_llm,
        embeddings=embeddings
    )

    scores = result.to_pandas().mean().to_dict()

    print("\n========== Evaluation Scores ==========\n")

    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")

    thresholds = load_thresholds()

    failed = False

    for metric, threshold in thresholds.items():

        if metric in scores and scores[metric] < threshold:

            print(
                f"❌ {metric} below threshold "
                f"({scores[metric]:.3f} < {threshold})"
            )

            failed = True

    if failed:
        raise SystemExit(
            "Build failed because one or more evaluation scores "
            "are below the configured thresholds."
        )

    print("\n✅ All evaluation scores passed the required thresholds.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run()