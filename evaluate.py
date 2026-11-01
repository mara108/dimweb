"""
Подсчёт метрик качества LLM-фильтра относительно эталонных ответов.

Запуск: python evaluate.py (из папки проекта, с заполненным .env).

Клиенты LLM создаются ТЕМ ЖЕ make_clients() и из того же .env, что и в
боевом пайплайне (generate_live.py) — поэтому метрики относятся именно к
той модели, которая реально подключена (сейчас YandexGPT через
LLM_OPEN_AI_*), а не к прежним захардкоженным клиентам. Ключи в коде
не хранятся.

Сравниваются: GPT-клиент из make_clients() сам по себе, GigaChat сам по
себе и AND-объединение с постфильтрами, словарём, оценкой уверенности и
разрешением конфликтов пар (то, что пользователь реально видит в
основном списке generate_diminutives()).

Формы с exists=null в эталоне (спорные/неподтверждённые случаи) исключаются
из подсчёта метрик — лучше честно сократить выборку, чем оценивать модель
по случаям, где у нас самих нет уверенного ответа.
"""

import json
import os
import time


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
                got = r.get("connotation")
                # merge_and при расхождении моделей отдаёт СПИСОК [gpt, gc];
                # засчитываем, если хотя бы одна оценка допустима
                got_list = got if isinstance(got, list) else [got]
                if any(c in accepted for c in got_list):
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
    header = f"{'Модель':<48} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Коннотация':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        conn = f"{m['connotation_accuracy']:.2f}" if m['connotation_accuracy'] == m['connotation_accuracy'] else "н/д"
        print(f"{m['model']:<48} {m['precision']:>10.2f} {m['recall']:>10.2f} "
              f"{m['f1']:>8.2f} {conn:>12}")


def short_model_name(name: str) -> str:
    """Для YandexGPT имя модели — URI вида gpt://<id_каталога>/yandexgpt/latest;
    ID каталога в отчёт не выводим."""
    if name and name.startswith("gpt://"):
        return "/".join(name.split("/")[3:])
    return name or "?"


def collect(label: str, fn, words: list[str]) -> dict[str, list[dict]]:
    """Прогоняет fn по словам; сбой на одном слове (сеть, лимиты) не должен
    ронять весь прогон, а платные вызовы не стоит повторять с нуля."""
    results = {}
    for i, w in enumerate(words, 1):
        t = time.time()
        try:
            results[w] = fn(w)
            print(f"  [{i}/{len(words)}] {label}: «{w}» — {time.time() - t:.1f} с", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(words)}] {label}: «{w}» — СБОЙ {type(e).__name__}: {str(e)[:120]}", flush=True)
    return results


def count_none(results: dict[str, list[dict]]) -> int:
    """Вердикты exists=None: ответ модели не удалось разобрать (обрезан,
    невалидный JSON). Если их много — метрики занижены из-за формата, а не
    из-за качества модели."""
    return sum(1 for res in results.values() for r in res if r["exists"] is None)


def run_pipeline(word: str, gpt_res: list[dict], gc_res: list[dict]) -> list[dict]:
    """Те же шаги, что в generate_diminutives(): AND -> постфильтр -> словарь
    -> оценка уверенности -> разрешение конфликтов пар. Формы, перенесённые
    в needs_review из-за конфликта «ь», в основном списке не показываются,
    поэтому для метрик считаются непринятыми (exists=False)."""
    merged = merge_and(gpt_res, gc_res)
    postfiltered = apply_palatalization_postfilter(word, merged)
    final = cross_check_with_dictionary(word, postfiltered)
    final = attach_confidence(final)

    accepted = [r for r in final if r["exists"] is True]
    _, moved = resolve_suffix_pair_conflicts(accepted)
    moved_forms = {r["form"] for r in moved}
    for r in final:
        if r["form"] in moved_forms:
            r["exists"] = False
    return final


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    from generator import gen_candidates
    from filter_gpt import filter_candidates as filter_gpt_candidates
    from filter_gigachat import filter_candidates as filter_gigachat_candidates
    from generate_live import (
        make_clients, merge_and, attach_confidence,
    )
    from palatalization_postfilter import apply_palatalization_postfilter
    from opencorpora_check import cross_check_with_dictionary
    from confidence_score import resolve_suffix_pair_conflicts

    gigachat_client, gpt_llm = make_clients()
    gpt_name = short_model_name(os.getenv("LLM_OPEN_AI_MODEL_NAME"))
    gc_name = short_model_name(os.getenv("LLM_GIGA_CHAT_MODEL_NAME"))
    print(f"GPT-клиент: {gpt_name} | GigaChat: {gc_name}\n")

    ground_truth = load_ground_truth("ground_truth_merged.json")
    test_words = list(ground_truth.keys())
    candidates = {w: gen_candidates(w) for w in test_words}

    print("Запросы к GPT-клиенту:")
    gpt_results = collect(gpt_name, lambda w: filter_gpt_candidates(gpt_llm, w, candidates[w]), test_words)
    print("Запросы к GigaChat:")
    gc_results = collect(gc_name, lambda w: filter_gigachat_candidates(gigachat_client, w, candidates[w]), test_words)

    # Все три строки считаем на ОДНОМ наборе слов — иначе сбой на отдельных
    # словах сделал бы сравнение несправедливым.
    common = [w for w in test_words if w in gpt_results and w in gc_results]
    skipped = [w for w in test_words if w not in common]
    print(f"\nСлов в оценке: {len(common)} из {len(test_words)}" +
          (f" (пропущены из-за сбоев: {', '.join(skipped)})" if skipped else ""))
    if not common:
        print("Ни одно слово не обработано обеими моделями — проверьте ключи, сеть и лимиты.")
        raise SystemExit(1)

    gpt_results = {w: gpt_results[w] for w in common}
    gc_results = {w: gc_results[w] for w in common}
    print(f"Вердиктов без разбора (exists=None): GPT-клиент {count_none(gpt_results)}, "
          f"GigaChat {count_none(gc_results)}")

    and_results = {w: run_pipeline(w, gpt_results[w], gc_results[w]) for w in common}

    gpt_metrics = evaluate_model(gpt_name, gpt_results, ground_truth)
    gc_metrics = evaluate_model(f"{gc_name} (сама по себе)", gc_results, ground_truth)
    and_metrics = evaluate_model(f"AND: {gpt_name} + {gc_name} (полный пайплайн)", and_results, ground_truth)

    compare_models(gpt_metrics, gc_metrics, and_metrics)