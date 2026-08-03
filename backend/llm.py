# llm.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.prompts import load_prompt
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)


def generate_answer(query, context):

    prompt_cfg = load_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_cfg["system"]),
        ("user", prompt_cfg["user"])
    ])

    chain = prompt | llm

    response = chain.invoke({
        "query": query,
        "context": context
    })

    return response.content