import json
import yaml
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)

# Import from root files
from hybrid import hybrid_search
from context import build_context
from llm import generate_answer


# ---------------- LOAD DATA ----------------

def load_dataset(path="../data/golden_dataset.json"):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_thresholds(path="thresholds.yaml"):

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



# ---------------- BUILD EVALUATION DATA ----------------

def build_eval_dataframe():

    data = load_dataset()

    rows = []

    for item in data:

        question = item["question"]
        ground_truth = item["ground_truth"]


        # Retrieve documents from Hybrid RAG
        docs = hybrid_search(
            query=question,
            k=5
        )


        # RAGAS requires List[str]
        contexts = [
            doc.page_content
            for doc in docs
        ]


        # Context with citations for LLM
        context_text = build_context(docs)


        # Generate RAG answer
        answer = generate_answer(
            query=question,
            context=context_text
        )


        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts
            }
        )


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


    scores = (
        result
        .to_pandas()
        .mean(numeric_only=True)
        .to_dict()
    )


    print("\n========== RAG EVALUATION SCORES ==========\n")


    for metric, score in scores.items():

        print(
            f"{metric}: {score:.3f}"
        )


    thresholds = load_thresholds()


    print("\n========== THRESHOLD CHECK ==========\n")


    failed = False


    for metric, threshold in thresholds.items():

        score = scores.get(metric, 0)


        if score < threshold:

            print(
                f"❌ {metric}: {score:.3f} < {threshold}"
            )

            failed = True

        else:

            print(
                f"✅ {metric}: {score:.3f} >= {threshold}"
            )


    if failed:

        raise SystemExit(
            "❌ RAG Evaluation Failed"
        )


    print(
        "\n✅ All metrics passed"
    )



if __name__ == "__main__":

    run()