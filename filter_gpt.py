"""
Обёртка LLM-фильтра поверх gpt-4o-mini (через LangChain ChatOpenAI).
Вся логика промпта/валидации — в llm_filter_core.py, здесь только
адаптация конкретного API под интерфейс call_fn(system, user) -> str.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llm_filter_core import filter_candidates as _filter_candidates


def make_call_fn(llm):
    """Адаптирует langchain llm.invoke под call_fn(system_prompt, user_prompt) -> str."""
    def call_fn(system_prompt: str, user_prompt: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        return response.content
    return call_fn


def filter_candidates(llm, lemma: str, candidates: list[dict], max_retries: int = 2) -> list[dict]:
    call_fn = make_call_fn(llm)
    return _filter_candidates(call_fn, lemma, candidates, max_retries=max_retries)


if __name__ == "__main__":
    import os
    from generator import gen_candidates
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=os.environ.get("COURSE_API_KEY"),
        temperature=0,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    for word in ["актриса", "рука", "дождь"]:
        candidates = gen_candidates(word)
        filtered = filter_candidates(llm, word, candidates)
        print(f"\n{word}:")
        for r in filtered:
            status = "✓" if r["exists"] else ("✗" if r["exists"] is False else "?")
            print(f"  {status} {r['form']:<16} connotation={r['connotation']}")