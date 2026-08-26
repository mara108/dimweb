"""
Извлечение эталонных ответов (ground truth) из реального эксперимента
(diminutives_experiment_full.json) — опрос респондентов, которых просили
самостоятельно образовать уменьшительные/ласкательные/пренебрежительные
формы от заданных существительных.

Критерии отбора (по требованию пользователя):
  - только группа "students" (студенты 1-го курса, отделение «Журналистика») —
    самая надёжная группа респондентов из трёх (в отличие от 2-го и 6-го
    классов, где выше риск случайных/ошибочных/шуточных ответов детей);
  - только формы с частотой > 1 (то есть их независимо друг от друга
    предложили минимум 2 респондента) — единичный ответ может быть
    опиской, идиосинкразией или ошибкой одного человека, а не реальным
    общеупотребимым словом.

Один и тот же респондентский набор данных даёт по каждому базовому слову
три категории (size/affection/pejorative — по терминологии авторов
исследования). Мы сопоставляем их с нашей схемой коннотаций:
    size       -> нейтральное
    affection  -> ласкательное
    pejorative -> пренебрежительное

Если одна и та же форма встретилась в нескольких категориях (респонденты
разошлись во мнении, куда её отнести — типичная ситуация для диминутивов
с реально смешанной/контекстной коннотацией), в качестве основной
коннотации берём категорию с наибольшей суммарной частотой, а расхождение
явно фиксируем в поле "note" вместо того чтобы его прятать.
"""

import json

CATEGORY_TO_CONNOTATION = {
    "size": "нейтральное",
    "affection": "ласкательное",
    "pejorative": "пренебрежительное",
}


def load_experiment(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_ground_truth(
    experiment_data: dict,
    respondent_group: str = "students",
    min_freq: int = 2,
    connotation_share_threshold: float = 0.20,
) -> dict:
    """
    Возвращает словарь в том же формате, что и ground_truth.json:
    {lemma: {form: {"exists": true, "connotation": ..., "accepted_connotations": [...],
                     "source": ...}, ...}, ...}

    "connotation" — категория-лидер (для удобного отображения).
    "accepted_connotations" — все категории, набравшие долю >= connotation_share_threshold
    от общего числа упоминаний ЭТОЙ формы (across категорий). Респонденты часто
    расходятся во мнении (диминутив контекстно многозначен), поэтому при подсчёте
    метрик модель не должна штрафоваться за попадание в весомое "второе мнение" —
    только категория-лидер была бы слишком строгим критерием.
    """
    group_data = experiment_data["data"].get(respondent_group)
    if group_data is None:
        raise ValueError(f"Группа респондентов '{respondent_group}' не найдена в файле.")

    result = {}

    for lemma, categories in group_data.items():
        # форма -> список (коннотация, частота) по всем категориям, где она встретилась
        form_occurrences: dict[str, list[tuple[str, int]]] = {}

        for category_key, forms in categories.items():
            connotation = CATEGORY_TO_CONNOTATION.get(category_key)
            if connotation is None:
                continue  # неизвестная категория — пропускаем, а не падаем
            for form, count in forms.items():
                if count > min_freq - 1:  # то есть count >= min_freq, при min_freq=2 это count>1
                    form_occurrences.setdefault(form, []).append((connotation, count))

        if not form_occurrences:
            continue

        word_ground_truth = {}
        for form, occurrences in form_occurrences.items():
            occurrences.sort(key=lambda x: -x[1])
            best_connotation, best_count = occurrences[0]
            total_mentions = sum(count for _, count in occurrences)

            accepted = [
                conn for conn, count in occurrences
                if count / total_mentions >= connotation_share_threshold
            ]

            note = (f"эксперимент, группа '{respondent_group}', "
                     f"частота={best_count} ({best_connotation})")
            if len(occurrences) > 1:
                others = ", ".join(f"{conn}={cnt}" for conn, cnt in occurrences[1:])
                note += f"; расхождение респондентов — также встречается как {others}"
                if len(accepted) > 1:
                    note += f"; допустимые категории (>={connotation_share_threshold:.0%}): {accepted}"

            word_ground_truth[form] = {
                "exists": True,
                "connotation": best_connotation,
                "accepted_connotations": accepted,
                "source": note,
            }

        result[lemma] = word_ground_truth

    return result


def merge_ground_truth(*sources: dict, on_conflict: str = "keep_first") -> tuple[dict, list[dict]]:
    """
    Объединяет несколько словарей ground truth. Если одна и та же лемма+форма
    встречается в разных источниках с РАЗНЫМ exists/connotation — это конфликт,
    который не разрешается молча: возвращаем список конфликтов отдельно,
    чтобы решение принял человек, а не код.

    on_conflict: "keep_first" — при конфликте оставляет версию из первого
    источника (последующие источники не перезаписывают существующую лемму+форму
    молча), но конфликт всё равно логируется.
    """
    merged: dict = {}
    conflicts = []

    for source_idx, source in enumerate(sources):
        for lemma, forms in source.items():
            merged.setdefault(lemma, {})
            for form, entry in forms.items():
                if form not in merged[lemma]:
                    merged[lemma][form] = entry
                else:
                    existing = merged[lemma][form]
                    existing_accepted = set(existing.get("accepted_connotations") or [existing.get("connotation")])
                    new_accepted = set(entry.get("accepted_connotations") or [entry.get("connotation")])
                    if (existing.get("exists") != entry.get("exists")
                            or existing_accepted != new_accepted):
                        conflicts.append({
                            "lemma": lemma,
                            "form": form,
                            "kept": existing,
                            "discarded_from_source_index": source_idx,
                            "discarded": entry,
                        })
                    # keep_first: ничего не перезаписываем

    return merged, conflicts


if __name__ == "__main__":
    import sys

    experiment_path = sys.argv[1] if len(sys.argv) > 1 else "diminutives_experiment_full.json"
    existing_gt_path = sys.argv[2] if len(sys.argv) > 2 else "ground_truth.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "ground_truth_merged.json"

    experiment_data = load_experiment(experiment_path)
    extracted = extract_ground_truth(experiment_data, respondent_group="students", min_freq=2)

    print(f"Извлечено лемм из эксперимента: {len(extracted)}")
    total_forms = sum(len(forms) for forms in extracted.values())
    print(f"Всего форм (частота > 1): {total_forms}\n")

    for lemma, forms in extracted.items():
        print(f"{lemma}:")
        for form, entry in forms.items():
            print(f"  {form:<18} {entry['connotation']:<20} {entry['source']}")
        print()

    try:
        with open(existing_gt_path, encoding="utf-8") as f:
            existing_gt = json.load(f)
    except FileNotFoundError:
        existing_gt = {}
        print(f"Файл {existing_gt_path} не найден, объединяю только с пустым словарём.")

    merged, conflicts = merge_ground_truth(existing_gt, extracted)

    if conflicts:
        print(f"\n{'=' * 50}")
        print(f"НАЙДЕНЫ КОНФЛИКТЫ ({len(conflicts)}) — требуют ручного решения:")
        print(f"{'=' * 50}")
        for c in conflicts:
            print(f"  [{c['lemma']}] {c['form']}:")
            print(f"    оставлено:  {c['kept']}")
            print(f"    отброшено:  {c['discarded']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nИтоговый объединённый файл сохранён в {out_path}")
    print(f"Всего лемм в объединённом файле: {len(merged)}")
