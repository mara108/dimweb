"""
Общее ядро LLM-фильтрации кандидатов-диминутивов.

Вынесено отдельно от конкретного провайдера (gpt-4o-mini / GigaChat / ...),
чтобы при сравнении моделей единственной переменной была сама модель,
а не ещё и разный промпт или разная логика валидации ответа.

Конкретные обёртки (filter_gpt.py, filter_gigachat.py) реализуют только
call_fn(system_prompt, user_prompt) -> raw_text_response и вызывают
filter_candidates() отсюда.
"""

import json
import re


SYSTEM_PROMPT = """Ты — эксперт по русской морфологии и словообразованию,
специализируешься на уменьшительно-ласкательных формах существительных
(диминутивах). Твоя задача — из списка автоматически сгенерированных
кандидатов отобрать только те формы, которые реально существуют и
употребляются в русском языке (в живой речи, разговорном стиле,
художественной литературе — не обязательно в словаре литературной нормы,
просторечные и диалектные формы тоже считаются существующими).

Обрати особое внимание на чередования согласных перед суффиксом
(рука -> руч-, нога -> нож-, муха -> муш-): формы БЕЗ нужного чередования
обычно не существуют (например, "рукочка" неверно, правильно "ручечка"
или "ручка"), а формы С чередованием, в том числе неожиданные на первый
взгляд, часто вполне реальны и употребимы (ручонка, ручища — не путай
разговорность/просторечность с несуществованием).

Отвечай ТОЛЬКО в формате JSON, без какого-либо текста до или после.
Не используй markdown-разметку (никаких ```json блоков)."""


FEW_SHOT_EXAMPLES = [
    {"lemma": "доска", "form": "дощечка", "exists": True, "connotation": "ласкательное"},
    {"lemma": "край", "form": "краешек", "exists": True, "connotation": "ласкательное"},
    {"lemma": "дождь", "form": "дождик", "exists": True, "connotation": "нейтральное"},
    {"lemma": "дождь", "form": "дождчик", "exists": False, "connotation": None},
    {"lemma": "актриса", "form": "актриска", "exists": True, "connotation": "пренебрежительное"},
    {"lemma": "актриса", "form": "актрисулька", "exists": True, "connotation": "пренебрежительное"},
    # ниже — примеры, добавленные после разбора конкретных ошибок фильтра
    # на паре "рука": модель путала формы с чередованием и без
    {"lemma": "рука", "form": "рукочка", "exists": False, "connotation": None},
    {"lemma": "рука", "form": "ручечка", "exists": True, "connotation": "ласкательное"},
    {"lemma": "рука", "form": "ручонка", "exists": True, "connotation": "пренебрежительное"},
    {"lemma": "рука", "form": "ручища", "exists": True, "connotation": "нейтральное"},
]


def build_user_prompt(lemma: str, candidates: list[dict]) -> str:
    """
    candidates — вывод generator.gen_candidates(lemma):
    [{"form": ..., "suffix": ..., "connotation_hint": ..., "note": ...}, ...]
    """
    examples_block = "\n".join(
        f'  {{"лемма": "{e["lemma"]}", "форма": "{e["form"]}", '
        f'"существует": {str(e["exists"]).lower()}, '
        f'"коннотация": {json.dumps(e["connotation"], ensure_ascii=False)}}}'
        for e in FEW_SHOT_EXAMPLES
    )

    candidates_block = "\n".join(
        f'  - "{c["form"]}" (предварительная коннотация по суффиксу: {c["connotation_hint"]})'
        for c in candidates
    )

    return f"""Примеры корректной разметки (для калибровки, не связаны с текущим словом):
{examples_block}

Существительное: "{lemma}"

Кандидаты диминутивных форм (сгенерированы автоматически правилами,
среди них есть реально существующие и есть некорректные/несуществующие):
{candidates_block}

Для КАЖДОГО кандидата из списка выше верни объект со следующими полями:
- "форма": сама форма (строка, точно как в списке кандидатов, без изменений)
- "существует": true/false — существует ли эта форма в русском языке
- "коннотация": одно из "нейтральное", "ласкательное", "пренебрежительное"
  (если существует), или null (если существует=false).
  Предварительная коннотация по суффиксу — это только подсказка,
  ты можешь её изменить, если для конкретного слова оттенок другой
  (один и тот же суффикс может давать разную окраску в зависимости от базы).

Верни JSON-объект строго такого вида (ключ "результаты" обязателен):
{{"результаты": [
  {{"форма": "...", "существует": true, "коннотация": "..."}},
  ...
]}}"""


def strip_code_fences(text: str) -> str:
    """На случай если модель обернула ответ в ```json ... ```."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


VALID_CONNOTATIONS = {"нейтральное", "ласкательное", "пренебрежительное"}


def validate_and_normalize(parsed: dict, original_forms: set[str]) -> list[dict]:
    """
    Не доверяем структуре ответа вслепую (ни json_object, ни промпт-based
    JSON без строгой схемы этого не гарантируют). Проверяем каждое поле
    и отбрасываем некорректные записи вместо падения всего пайплайна
    на одном плохом ответе.
    """
    results = parsed.get("результаты")
    if not isinstance(results, list):
        raise ValueError(f"Ожидался список в поле 'результаты', получено: {type(results)}")

    validated = []
    for item in results:
        if not isinstance(item, dict):
            continue
        form = item.get("форма")
        exists = item.get("существует")
        connotation = item.get("коннотация")

        if not isinstance(form, str) or form not in original_forms:
            continue  # модель могла придумать форму не из списка — отбрасываем
        if not isinstance(exists, bool):
            continue

        if exists and connotation not in VALID_CONNOTATIONS:
            connotation = None  # модель дала невалидную коннотацию — не выдумываем
        if not exists:
            connotation = None

        validated.append({
            "form": form,
            "exists": exists,
            "connotation": connotation,
        })

    return validated


def filter_candidates(call_fn, lemma: str, candidates: list[dict], max_retries: int = 2) -> list[dict]:
    """
    Провайдеро-независимая логика фильтрации.

    call_fn: callable(system_prompt: str, user_prompt: str) -> str
        Обёртка над конкретным API (OpenAI через LangChain, GigaChat через
        requests, ...), которая просто возвращает сырой текстовый ответ модели.

    Возвращает список:
    [{"form": ..., "suffix": ..., "exists": ..., "connotation": ..., "note": ...}, ...]

    Формы, которые не удалось корректно оценить (сбой парсинга после всех
    попыток), возвращаются с exists=None — явный сигнал "не проверено",
    а не тихая потеря данных.
    """
    if not candidates:
        return []

    original_forms = {c["form"] for c in candidates}
    user_prompt = build_user_prompt(lemma, candidates)

    last_error = None
    validated = None
    for attempt in range(max_retries + 1):
        try:
            raw_response = call_fn(SYSTEM_PROMPT, user_prompt)
            raw = strip_code_fences(raw_response)
            parsed = json.loads(raw)
            validated = validate_and_normalize(parsed, original_forms)
            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            validated = None

    if validated is None and last_error is not None:
        print(f"[llm_filter_core] Не удалось получить корректный ответ для '{lemma}' "
              f"после {max_retries + 1} попыток: {last_error}")

    by_form = {v["form"]: v for v in validated} if validated else {}

    result = []
    for c in candidates:
        llm_result = by_form.get(c["form"])
        result.append({
            "form": c["form"],
            "suffix": c["suffix"],
            "exists": llm_result["exists"] if llm_result else None,
            "connotation": llm_result["connotation"] if llm_result else None,
            "note": c["note"],
        })

    return result