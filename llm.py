from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from prompts import load_prompt

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

def generate_answer(query, context):
    prompt_cfg = load_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_cfg["system"]),
        ("user", prompt_cfg["user"])
    ])

    messages = prompt.format_messages(
        query=query,
        context=context
    )

    response = llm.invoke(messages)

    return response.content