"""
Бесплатная сборка базового словаря диминутивов — БЕЗ обращения к LLM.

Пайплайн: gen_candidates() -> оставляем только in_opencorpora=True.

Никаких API-вызовов, никакой сети (кроме локального pymorphy2) — можно
прогонять хоть на всех 1000+ леммах за секунды, а не часы, и без всякой
стоимости. За это платим полнотой: OpenCorpora покрывает опубликованные
тексты, а не живую разговорную речь, поэтому многие реальные разговорные
диминутивы (например, "кроваточка" — подтверждено внешним поиском, но не
входит в словарь OpenCorpora) в этот список НЕ попадут. Это осознанный
компромисс: этот словарь — быстрый бесплатный черновой охват для раздела
"Словарь" приложения, а не исчерпывающий список. Для конкретного слова
пользователь может получить полный список через раздел "Генерация"
(build_dictionary.py с LLM-фильтрацией, живой запрос на лету).

ВАЖНО: здесь НЕ применяется palatalization_postfilter — он создан для
исправления ошибок LLM, а не для того, чтобы перебивать прямое
подтверждение словаря. Если OpenCorpora сказала "да" — это надёжнее
эвристики о чередовании, отменять его не нужно.

Коннотация в этом словаре — это connotation_hint из generator.py
(эвристика по суффиксу из статичной таблицы), а НЕ подтверждённая LLM
коннотация. Это слабее, чем в разделе "Генерация", и стоит явно показывать
пользователю с пометкой "предварительно" в интерфейсе.
"""

import json
import os

from generator import gen_candidates


def load_frequency_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return [e["lemma"] for e in entries]


def build_free_dictionary(frequency_list_path: str, out_path: str, limit: int | None = None) -> None:
    lemmas = load_frequency_list(frequency_list_path)
    if limit:
        lemmas = lemmas[:limit]

    dictionary = {}
    errors = []

    for i, lemma in enumerate(lemmas, 1):
        try:
            candidates = gen_candidates(lemma)
            accepted = [c for c in candidates if c["in_opencorpora"]]
            dictionary[lemma] = {
                "status": "ok",
                "forms": [
                    {
                        "form": c["form"],
                        "suffix": c["suffix"],
                        "connotation_hint": c["connotation_hint"],  # эвристика, не подтверждено LLM
                    }
                    for c in accepted
                ],
                "total_candidates": len(candidates),
                "accepted_count": len(accepted),
            }
        except Exception as e:
            errors.append(lemma)
            dictionary[lemma] = {"status": "error", "error": str(e), "forms": []}

        if i % 100 == 0 or i == len(lemmas):
            print(f"[{i}/{len(lemmas)}] обработано")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    total_forms = sum(len(v["forms"]) for v in dictionary.values())
    words_with_zero_forms = sum(1 for v in dictionary.values() if v["status"] == "ok" and not v["forms"])

    print(f"\nГотово. Сохранено в {out_path}")
    print(f"Всего лемм: {len(lemmas)}, ошибок: {len(errors)}")
    print(f"Всего подтверждённых форм: {total_forms}")
    print(f"Лемм без единой подтверждённой формы (только OpenCorpora не знает ни одной): {words_with_zero_forms}")
    if errors:
        print(f"Слова с ошибками: {errors[:20]}{'...' if len(errors) > 20 else ''}")


if __name__ == "__main__":
    FREQUENCY_LIST_PATH = "frequency_nouns.json"
    OUT_PATH = "dictionary_free.json"

    build_free_dictionary(FREQUENCY_LIST_PATH, OUT_PATH)
