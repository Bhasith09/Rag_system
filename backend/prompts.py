# prompts.py

import yaml

def load_prompt(path="prompttt/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["prompt"]