from groq import Groq
from backend.prompts import load_prompt
from dotenv import load_dotenv
import os

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, context):
    prompt_cfg=load_prompt()

    system=prompt_cfg["system"]
    user=prompt_cfg["user"].format(query=query, context=context)

    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}

        ],
        temperature=0.2
    )
    return response.choices[0].message.content