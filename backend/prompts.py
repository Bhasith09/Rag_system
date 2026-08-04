import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_PATH = os.path.join(BASE_DIR, "prompttt", "config.yaml")


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["prompt"]