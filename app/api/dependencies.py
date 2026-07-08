"""
dependencies.py — FastAPI dependency injectors.
retriever.py and llm_client.py already manage their own singletons
internally, so these injectors just expose the modules in a way that's
easy to swap/mock in tests.
"""
from app.core import citation, contradiction, llm_client, prompt_builder, retriever
from app.core.language import detect_and_translate


def get_retriever():
    return retriever


def get_prompt_builder():
    return prompt_builder


def get_llm_client():
    return llm_client


def get_citation_module():
    return citation


def get_contradiction_module():
    return contradiction


def get_language_module():
    return detect_and_translate
