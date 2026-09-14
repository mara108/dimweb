"""
Раздел "Генерация" — по одному слову, введённому пользователем "на лету",
возвращает полный список диминутивов с AND-стратегией (gpt-4o-mini И
GigaChat-2 должны согласиться) + постфильтр чередований + подтверждение
словарём OpenCorpora.

Пайплайн: gen_candidates -> [gpt-4o-mini И GigaChat-2 параллельно]
-> пересечение (AND) -> постфильтр чередований -> подтверждение словарём
(только восстановление ложноотклонённых) -> композитный score уверенности
(confidence_score.py) -> разделение на "принято" / "на экспертную проверку".

Самодостаточный модуль — вся логика (включая то, что раньше жило в
build_dictionary.py) здесь. Отдельный build_dictionary.py с пакетной
AND-LLM сборкой больше не нужен: "Словарь" (1000 частотных лемм) теперь
строится дёшево через build_dictionary_free.py на чистом generator.py
без LLM, а "Генерация" (это) — единственное место, где используется
дорогая двух-LLM проверка, и только по одному слову за раз, на лету по
запросу пользователя.
"""

import os

from generator import gen_candidates, get_noun_parse
from filter_gigachat import GigaChatClient, filter_candidates as filter_gigachat_candidates
from filter_gpt import filter_candidates as filter_gpt_candidates
from palatalization_postfilter import apply_palatalization_postfilter
from opencorpora_check import cross_check_with_dictionary
from confidence_score import compute_confidence, classify_for_pipeline, resolve_suffix_pair_conflicts


# ---------------------------------------------------------------------------
# Детекция "слово уже само похоже на диминутив" — против взрыва
# комбинаторики при попытке образовать диминутив от уже-диминутива
# (лепёшка -> 88 кандидатов, из которых почти все мусор).
# ---------------------------------------------------------------------------

# Окончания, совпадающие с суффиксами, которые сам generator.py использует
# для ОБРАЗОВАНИЯ диминутивов (см. gen_masc/gen_femn/gen_neut). Если лемма
# уже оканчивается на один из них — вероятно, это уже готовый диминутив.
#
# ВАЖНАЯ ОГОВОРКА: это ЭВРИСТИКА по совпадению окончания, не надёжный
# лингвистический критерий. Голое "-ка" сюда намеренно НЕ включено —
# слишком много лексикализованных слов оканчиваются на него, давно не
# воспринимаясь как диминутивы (ложка, кружка, кошка, белка, палка,
# бочка), и это давало массовые ложные срабатывания. Более длинные и
# специфичные окончания (-ечка, -очка, -ушка, -онька и т.п.) реже
# совпадают случайно, поэтому оставлены — но и они не застрахованы от
# ложных срабатываний полностью. Ответственность за то, чтобы не вводить
# уже готовый диминутив, частично перекладывается на пользователя через
# описание в интерфейсе приложения. Оставшиеся ложные срабатывания всё
# ещё можно обойти через force=True.
DIMINUTIVE_ENDINGS = {
    "masc": ["ичек", "ишко", "онок", "ёнок", "чик", "ик", "ок", "ек", "ёк", "ец"],
    "femn": ["онька", "енька", "ичка", "ечка", "очка", "ушка", "улька", "онка", "ёнка", "ёшка"],
    "neut": ["ышко", "ечко", "юшко", "ишко", "ице", "цо", "це", "ко"],
}


def detect_already_diminutive(word: str) -> dict | None:
    """
    Возвращает {"lemma": ..., "gender": ..., "matched_suffix": ...}, если
    лемма оканчивается на один из известных диминутивных суффиксов, иначе
    None. Проверка идёт от самых длинных/специфичных окончаний к самым
    коротким/частотным, чтобы, например, "ичка" не терялось за более
    общим "ка".

    len(lemma) > len(ending) + 1 — защита от вырожденных случаев (совсем
    короткая лемма, которая технически "оканчивается" на суффикс, но
    у которой перед ним нет содержательной основы).
    """
    parsed = get_noun_parse(word)
    lemma = parsed.normal_form
    gender = parsed.tag.gender

    endings = DIMINUTIVE_ENDINGS.get(gender)
    if not endings:
        return None

    for ending in sorted(endings, key=len, reverse=True):
        if lemma.endswith(ending) and len(lemma) > len(ending) + 1:
            return {"lemma": lemma, "gender": gender, "matched_suffix": ending}

    return None


def merge_and(gpt_result: list[dict], gc_result: list[dict]) -> list[dict]:
    """
    Принимаем форму только если ОБЕ модели согласны, что она существует.
    Ошибки моделей в основном разные (GigaChat путает чередования
    согласных, gpt-4o-mini просто более консервативна) — пересечение
    должно давать заметно более высокую точность, чем любая из моделей
    по отдельности.

    Коннотация: если обе модели согласны — берём её; если расходятся —
    сохраняем обе как список (решение за постфильтром/человеком дальше),
    вместо того чтобы произвольно отдавать предпочтение одной модели.

    Индивидуальные вердикты каждой модели (gpt_exists/gc_exists/
    gpt_connotation/gc_connotation) сохраняются ОТДЕЛЬНО от смёрженного
    both_exist/connotation — это нужно для compute_confidence()
    (confidence_score.py), которому важно различать "обе согласны" от
    "разногласие", а это теряется при схлопывании в единый bool.
    """
    gc_by_form = {r["form"]: r for r in gc_result}
    merged = []

    for gpt_r in gpt_result:
        form = gpt_r["form"]
        gc_r = gc_by_form.get(form)
        gc_exists = gc_r["exists"] if gc_r else None

        both_exist = bool(gpt_r["exists"]) and bool(gc_exists)

        if gpt_r["connotation"] and gc_r and gc_r["connotation"]:
            connotation = (gpt_r["connotation"] if gpt_r["connotation"] == gc_r["connotation"]
                           else [gpt_r["connotation"], gc_r["connotation"]])
        else:
            connotation = gpt_r["connotation"] or (gc_r["connotation"] if gc_r else None)

        merged.append({
            "form": form,
            "suffix": gpt_r["suffix"],
            "exists": both_exist,
            "connotation": connotation if both_exist else None,
            "note": gpt_r["note"],
            "gpt_exists": gpt_r["exists"],
            "gc_exists": gc_exists,
            "gpt_connotation": gpt_r["connotation"],
            "gc_connotation": gc_r["connotation"] if gc_r else None,
        })

    return merged


def attach_confidence(final_result: list[dict]) -> list[dict]:
    """
    Добавляет к каждой записи поля "confidence" (score/label/reasons из
    confidence_score.compute_confidence) и "review_channel"
    ("accepted"/"needs_review"/"rejected" из classify_for_pipeline).

    КРИТИЧЕСКИ ВАЖНО: НЕ меняет поле "exists" ни для одной записи.
    review_channel="accepted" присваивается ИСКЛЮЧИТЕЛЬНО на основе уже
    принятого пайплайном exists=True (результат AND + постфильтр +
    словарь, без изменений) — score используется только для того, чтобы
    разложить остальное (exists=False) по "needs_review"/"rejected",
    а не для того, чтобы задним числом включить в основной вывод формы,
    которые AND отклонил. См. classify_for_pipeline() в confidence_score.py.
    """
    for r in final_result:
        conf = compute_confidence(
            gpt_exists=r.get("gpt_exists"),
            gc_exists=r.get("gc_exists"),
            gpt_connotation=r.get("gpt_connotation"),
            gc_connotation=r.get("gc_connotation"),
            in_opencorpora=r.get("in_opencorpora", False),
            postfilter_rejected=r.get("postfilter_rejected", False),
        )
        r["confidence"] = conf
        r["review_channel"] = classify_for_pipeline(final_exists=r["exists"], confidence=conf)
    return final_result


def generate_diminutives(gigachat_client: GigaChatClient, gpt_llm, word: str, force: bool = False) -> dict:
    """
    Возвращает {"word": ..., "forms": [...только exists=True, как и
    раньше...], "needs_review": [...НОВОЕ: разногласие LLM, средний
    score, НЕ показывается пользователю по умолчанию...],
    "total_candidates": N, "accepted_count": M, "already_diminutive": bool,
    "message": str|None}.

    Двойной вызов LLM (gpt-4o-mini + GigaChat-2) на каждое слово —
    оправдано именно здесь: это единичный запрос пользователя, а не
    пакетная обработка тысяч лемм, так что удвоенная стоимость
    несущественна.

    force=True пропускает проверку detect_already_diminutive и генерирует
    в любом случае — нужно, потому что проверка эвристическая и может
    ошибочно сработать на словах вроде "ложка", "кошка" (см. docstring
    DIMINUTIVE_ENDINGS), для которых пользователь всё равно может хотеть
    получить двойной диминутив.
    """
    if not force:
        check = detect_already_diminutive(word)
        if check:
            message = (
                f"Слово «{word}» само оканчивается на диминутивный суффикс "
                f"(-{check['matched_suffix']}-) — похоже, это уже готовый диминутив, "
                f"а не базовая форма. Образование диминутива от диминутива обычно "
                f"даёт много искусственных, реально не употребляемых форм. "
                f"Если хотите всё же попробовать — повторите запрос с force=True."
            )
            return {
                "word": word,
                "forms": [],
                "needs_review": [],
                "total_candidates": 0,
                "accepted_count": 0,
                "already_diminutive": True,
                "message": message,
            }

    candidates = gen_candidates(word)
    if not candidates:
        return {
            "word": word, "forms": [], "needs_review": [],
            "total_candidates": 0, "accepted_count": 0,
            "already_diminutive": False, "message": None,
        }

    gpt_result = filter_gpt_candidates(gpt_llm, word, candidates)
    gc_result = filter_gigachat_candidates(gigachat_client, word, candidates)

    merged = merge_and(gpt_result, gc_result)
    postfiltered = apply_palatalization_postfilter(word, merged)
    final_result = cross_check_with_dictionary(word, postfiltered)
    final_result = attach_confidence(final_result)

    accepted = [r for r in final_result if r["exists"] is True]
    accepted, moved_to_review = resolve_suffix_pair_conflicts(accepted)
    needs_review = [r for r in final_result if r["review_channel"] == "needs_review"]
    needs_review += moved_to_review

    return {
        "word": word,
        "forms": accepted,
        "needs_review": needs_review,
        "total_candidates": len(candidates),
        "accepted_count": len(accepted),
        "already_diminutive": False,
        "message": None,
    }


def print_result(result: dict) -> None:
    """Человекочитаемый вывод для CLI/отладки."""
    word = result["word"]

    if result.get("already_diminutive"):
        print(f"\n{word}: {result['message']}")
        return

    forms = result["forms"]
    print(f"\n{word} ({result['accepted_count']}/{result['total_candidates']} прошли AND-фильтрацию):")
    if not forms:
        print("  Ни одной формы не подтверждено обеими моделями.")
    else:
        for f in sorted(forms, key=lambda x: x["form"]):
            connotation = f["connotation"]
            if isinstance(connotation, list):
                connotation = " / ".join(connotation)  # модели разошлись в коннотации
            conf = f.get("confidence")
            conf_str = f"  confidence={conf['label']}({conf['score']})" if conf else ""
            pair_note = f"  ⚠ {f['pair_conflict']}" if f.get("pair_conflict") else ""
            print(f"  {f['form']:<18} suffix={f['suffix']:<12} connotation={connotation}{conf_str}{pair_note}")

    # needs_review НЕ смешивается с основным списком — печатается отдельно
    # и только если непусто, чтобы не засорять обычный вывод.
    needs_review = result.get("needs_review") or []
    if needs_review:
        print(f"\n  На экспертную проверку ({len(needs_review)}, НЕ подтверждено автоматически):")
        for f in sorted(needs_review, key=lambda x: x["form"]):
            reasons = "; ".join(f["confidence"]["reasons"]) if f.get("confidence") else ""
            if f.get("pair_conflict"):
                reasons = f["pair_conflict"] if not reasons else f"{reasons}; {f['pair_conflict']}"
            score = f["confidence"]["score"] if f.get("confidence") else "?"
            print(f"    {f['form']:<18} score={score:<3} — {reasons}")


def make_clients() -> tuple[GigaChatClient, object]:
    """Создаёт обоих клиентов один раз — переиспользуйте между вызовами
    generate_diminutives(), не создавайте заново на каждое слово."""
    from langchain_openai import ChatOpenAI

    gigachat_client = GigaChatClient(
        authorization_key="MDE5ZjE0NzgtZDQxNy03MzZkLTgzZDQtYTQyZjViMWE1MWExOjFhOGYxZTI0LTE2MzMtNDMzNS1hNzdhLTMyZjdlMmMwMGRlZA==",
        model="GigaChat-2",
    )
    gpt_llm = ChatOpenAI(
        api_key="sk-dZ_Sgbz7nnyY8Nc8KdCzcg",
        temperature=0,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return gigachat_client, gpt_llm


if __name__ == "__main__":
    import sys

    gigachat_client, gpt_llm = make_clients()

    running_in_jupyter = "ipykernel" in sys.modules
    if running_in_jupyter:
        # в Jupyter удобнее вызывать напрямую:
        # result = generate_diminutives(gigachat_client, gpt_llm, "актриса")
        # print_result(result)
        for word in ["актриса", "рука"]:
            result = generate_diminutives(gigachat_client, gpt_llm, word)
            print_result(result)
    else:
        while True:
            word = input("\nВведите существительное (или пусто для выхода): ").strip()
            if not word:
                break
            try:
                result = generate_diminutives(gigachat_client, gpt_llm, word)
                print_result(result)
            except Exception as e:
                print(f"Ошибка при обработке '{word}': {e}")
