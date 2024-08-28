from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import dotenv
import os

dotenv.load_dotenv()


def load_groq_model() -> ChatGroq:
    groq_model = ChatGroq(
        model = "Llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1024
    )
    return groq_model

def load_openai_model() -> ChatOpenAI:
    openai_model = ChatOpenAI(
        model = "gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
        max_tokens=1024
    )
    return openai_model
