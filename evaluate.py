"""
Подсчёт метрик качества LLM-фильтра относительно эталонных ответов.

Это самостоятельный скрипт для проверки: python evaluate.py.
Запуск сравнивает gpt-4o-mini, GigaChat-2-Max и их AND-объединение
(полный пайплайн, как в generate_live.py) на ground_truth_merged.json —
код внизу файла (if __name__ == "__main__") уже готов к запуску,
никаких импортов из самого evaluate.py делать не нужно.

Формы с exists=null в эталоне (спорные/неподтверждённые случаи) исключаются
из подсчёта метрик — лучше честно сократить выборку, чем оценивать модель
по случаям, где у нас самих нет уверенного ответа.
"""

import json


def load_ground_truth(path: str) -> dict[str, dict[str, dict]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(
    model_name: str,
    model_results: dict[str, list[dict]],
    ground_truth: dict[str, dict[str, dict]],
    verbose: bool = True,
) -> dict:
    """
    model_results: {lemma: [{"form":..., "exists":..., "connotation":...}, ...], ...}
        — вывод filter_candidates() для каждого слова из ground_truth.

    Возвращает словарь с метриками; если verbose=True, дополнительно печатает
    отчёт и построчный разбор расхождений.
    """
    tp = fp = fn = tn = 0
    skipped_uncertain = 0
    skipped_not_in_gt = 0
    connotation_correct = 0
    connotation_total = 0
    mismatches = []
    connotation_mismatches = []

    for lemma, results in model_results.items():
        gt_for_word = ground_truth.get(lemma, {})
        for r in results:
            form = r["form"]
            gt_entry = gt_for_word.get(form)

            if gt_entry is None:
                skipped_not_in_gt += 1
                continue
            gt_exists = gt_entry["exists"]
            if gt_exists is None:
                skipped_uncertain += 1
                continue

            pred_exists = r["exists"]

            if gt_exists and pred_exists:
                tp += 1
            elif gt_exists and not pred_exists:
                fn += 1
                mismatches.append((lemma, form, "ожидалось exists=True, модель дала False/None"))
            elif not gt_exists and pred_exists:
                fp += 1
                mismatches.append((lemma, form, "ожидалось exists=False, модель дала True"))
            else:
                tn += 1

            # коннотацию проверяем только там, где обе стороны согласны, что форма существует
            if gt_exists and pred_exists and gt_entry.get("connotation"):
                # accepted_connotations — новое поле (из experiment-данных с
                # распределением мнений респондентов); для старых
                # вручную заполненных записей его нет — используем список
                # из одной "connotation" как fallback, сохраняя обратную
                # совместимость без правки всех прежних записей.
                accepted = set(gt_entry.get("accepted_connotations") or [gt_entry["connotation"]])

                connotation_total += 1
                if r.get("connotation") in accepted:
                    connotation_correct += 1
                else:
                    connotation_mismatches.append(
                        (lemma, form, gt_entry["connotation"], r.get("connotation"), accepted)
                    )

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    connotation_accuracy = connotation_correct / connotation_total if connotation_total else float("nan")

    metrics = {
        "model": model_name,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "connotation_accuracy": connotation_accuracy,
        "connotation_total": connotation_total,
        "skipped_uncertain": skipped_uncertain,
        "skipped_not_in_gt": skipped_not_in_gt,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Модель: {model_name}")
        print(f"{'=' * 50}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision (существование): {precision:.2f}")
        print(f"  Recall (существование):    {recall:.2f}")
        print(f"  F1:                        {f1:.2f}")
        print(f"  Accuracy:                  {accuracy:.2f}")
        if connotation_total:
            print(f"  Точность коннотации (на TP): {connotation_accuracy:.2f} "
                  f"({connotation_correct}/{connotation_total})")
        print(f"  Пропущено спорных (exists=null в эталоне): {skipped_uncertain}")
        if skipped_not_in_gt:
            print(f"  Пропущено форм не из эталона: {skipped_not_in_gt}")

        if mismatches:
            print(f"\n  Ошибки по exists ({len(mismatches)}):")
            for lemma, form, desc in mismatches:
                print(f"    [{lemma}] {form}: {desc}")

        if connotation_mismatches:
            print(f"\n  Ошибки по коннотации ({len(connotation_mismatches)}):")
            for lemma, form, expected, got, accepted in connotation_mismatches:
                print(f"    [{lemma}] {form}: получено '{got}', допустимо любое из {sorted(accepted)}")

    return metrics


def compare_models(*metrics_list: dict) -> None:
    """Печатает сводную таблицу метрик нескольких моделей рядом."""
    print(f"\n{'=' * 70}")
    print("СВОДНОЕ СРАВНЕНИЕ")
    print(f"{'=' * 70}")
    header = f"{'Модель':<20} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Коннотация':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        conn = f"{m['connotation_accuracy']:.2f}" if m['connotation_accuracy'] == m['connotation_accuracy'] else "н/д"
        print(f"{m['model']:<20} {m['precision']:>10.2f} {m['recall']:>10.2f} "
              f"{m['f1']:>8.2f} {conn:>12}")


if __name__ == "__main__":
    import os
    from generator import gen_candidates
    from filter_gpt import filter_candidates as filter_gpt_candidates
    from filter_gigachat import GigaChatClient, filter_candidates as filter_gigachat_candidates
    from langchain_openai import ChatOpenAI
    from generate_live import merge_and
    from palatalization_postfilter import apply_palatalization_postfilter
    from opencorpora_check import cross_check_with_dictionary

    ground_truth = load_ground_truth("ground_truth_merged.json")
    test_words = list(ground_truth.keys())

    llm = ChatOpenAI(
        api_key="sk-dZ_Sgbz7nnyY8Nc8KdCzcg",
        temperature=0,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    gpt_results = {w: filter_gpt_candidates(llm, w, gen_candidates(w)) for w in test_words}
    gpt_metrics = evaluate_model("gpt-4o-mini", gpt_results, ground_truth)

    # GigaChat-2-Max сам по себе (без AND, без постфильтров) — прямое
    # сравнение с прежним GigaChat-2 на тех же условиях
    max_client = GigaChatClient("MDE5ZjE0NzgtZDQxNy03MzZkLTgzZDQtYTQyZjViMWE1MWExOjFhOGYxZTI0LTE2MzMtNDMzNS1hNzdhLTMyZjdlMmMwMGRlZA==", model="GigaChat-2-Max")
    max_results = {w: filter_gigachat_candidates(max_client, w, gen_candidates(w)) for w in test_words}
    max_metrics = evaluate_model("GigaChat-2-Max (сама по себе)", max_results, ground_truth)

    # AND-объединение gpt-4o-mini + GigaChat-2-Max + постфильтры + OpenCorpora —
    # это ровно тот пайплайн, что реально используется в generate_live.py,
    # поэтому именно эта строка важнее всего для решения "менять ли модель по умолчанию"
    and_results = {}
    for w in test_words:
        merged = merge_and(gpt_results[w], max_results[w])
        postfiltered = apply_palatalization_postfilter(w, merged)
        final = cross_check_with_dictionary(w, postfiltered)
        and_results[w] = final
    and_metrics = evaluate_model("AND: gpt-4o-mini + GigaChat-2-Max (полный пайплайн)", and_results, ground_truth)

    compare_models(gpt_metrics, max_metrics, and_metrics)