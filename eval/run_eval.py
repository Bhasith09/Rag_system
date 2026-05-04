import json
import yaml
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
#my pipeline

import backend
from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer

#load the data

def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)
    
def load_thresholds(path="eval/thresholds.yaml"):
    with open(path ,"r") as f:
        return yaml.safe_load(f)
    
#build eval data

def build_eval_dataframe():
    data=load_dataset()
    rows=[]
    for item in data:
        question=item["question"]
        ground_truth=item["ground_truth"]

        docs=hybrid_search(question,k=5)

        #IMPORNTANT: ragas expects list of contexts

        contexts = docs if docs else []

        context_text=build_context(contexts)
        answer=generate_answer(question,context_text)

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts
            }
        )

    return pd.DataFrame(rows)

#Run Evaluation

def run():
    df=build_eval_dataframe()
    dataset=Dataset.from_pandas(df)  #convert to hugging face format bcz its required for ragas

    result=evaluate(dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_recall
                    ])
    scores=result.to_pandas().mean().to_dict()
    print("\n===Evaluation Scores===")
    for k,v in scores.items():
        print(f"{k}: {v:.3f}")

    #threshold chcek
    thresholds=load_thresholds()
    failed=False
    for metrics, threshold in thresholds.items():
        if scores[metrics]<threshold:
            print(f"❌ {metrics} below threshold: {scores[metrics]:.3f} < {threshold}")
            failed=True
    if failed:
        raise SystemExit("Build failed due to low evaluation scores")
    print("✅ All evaluation scores are above the thresholds")

if __name__=="__main__":
    run()

                    