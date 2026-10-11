from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def pick_llm(level: str):
    level = level.lower()

    if level == "low":
        llm = ChatOpenAI(
            model="gpt-5.6-luna",
            temperature=0,
        )
    elif level == "medium":
        llm = ChatOpenAI(
            model="gpt-5.6-terra",
            temperature=0,
        )
    elif level == "high":
        llm = ChatOpenAI(
            model="gpt-5.6-sol",
            temperature=0,
        )
    else:
        raise ValueError(f"Invalid model level: {level}")

    return llm
