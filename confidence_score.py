"""
Композитный score уверенности для кандидатов-диминутивов.

НЕ пытается извлечь "уверенность" из logprobs отдельной LLM — это уже
пробовали (filter_gpt_logprobs.py, отклонён): модель систематически
самоуверенна именно там, где ошибается (p≈1.00 на систематических
ошибках), так что logprob одной модели не несёт полезного сигнала.

Вместо этого комбинирует НЕЗАВИСИМЫЕ сигналы, уже существующие в
пайплайне и не сводящиеся друг к другу:
    1. Согласие/несогласие двух LLM (а не только факт прохождения AND).
    2. Подтверждение словарём OpenCorpora — самый надёжный источник
       (см. docstring opencorpora_check.py).
    3. Детерминированный постфильтр чередований — жёсткое лингвистическое
       правило, не вероятностное (palatalization_postfilter.py).
    4. Продуктивность суффикса из suffix_connotation_table.json.

Назначение: не заменить бинарный AND, а дать промежуточную метку для
случаев, которые сейчас пайплайн либо молча отбрасывает (одна LLM за,
другая против), либо принимает без оговорок (обе синхронно ошиблись).
Это ровно недостающее звено для "экспертной верификации лингвистом",
упомянутой в планах проекта, но не реализованной в коде.
"""

from typing import Optional

PRODUCTIVITY_WEIGHT = {
    "высокопродуктивен": 1.0,
    "продуктивен": 0.7,
    "эпизодически продуктивен": 0.3,
}


def _productivity_bonus(suffix_note_productivity: Optional[str]) -> int:
    """Небольшая поправка (не решающий фактор) — продуктивный суффикс
    сам по себе не доказывает существование КОНКРЕТНОЙ формы, но слегка
    повышает априорную вероятность по сравнению с редким/непродуктивным."""
    if not suffix_note_productivity:
        return 0
    for key, weight in PRODUCTIVITY_WEIGHT.items():
        if key in suffix_note_productivity:
            return round(weight * 10) - 5  # диапазон примерно [-2, +5]
    return 0


def compute_confidence(
    gpt_exists: Optional[bool],
    gc_exists: Optional[bool],
    gpt_connotation: Optional[str],
    gc_connotation: Optional[str],
    in_opencorpora: bool,
    postfilter_rejected: bool = False,
    suffix_productivity_note: Optional[str] = None,
) -> dict:
    """
    Возвращает {"score": 0-100, "label": "высокая"/"средняя"/"низкая",
    "reasons": [...]} — с явным объяснением, ЧТО именно повлияло на
    оценку. Объяснимость обязательна: это оценка для лингвиста, а не
    скрытая метрика для автоматического отсева.
    """
    reasons = []

    # Словарь — самый сильный независимый сигнал, перебивает
    # разногласие LLM (та же логика, что уже в cross_check_with_dictionary,
    # только явно выражена как число, а не бинарная перезапись).
    if in_opencorpora:
        if gpt_connotation or gc_connotation:
            score = 90
            reasons.append("форма подтверждена словарём OpenCorpora")
        else:
            # Словарь подтверждает СТРОКУ, но ни одна модель не смогла
            # приписать ей коннотацию как диминутиву ИМЕННО этой леммы —
            # похоже на омограф (реальный случай: "свинушка" — гриб,
            # а не диминутив "свиньи"; словарь его знает, но не в этом
            # смысле). Понижаем, а не отбрасываем совсем — словарь всё
            # же сильный сигнал, просто не абсолютный.
            score = 65
            reasons.append(
                "строка есть в словаре OpenCorpora, но ни одна модель не "
                "подтвердила коннотацию как диминутива именно этого слова "
                "— возможен омограф (слово с тем же написанием, но другим значением)"
            )
    else:
        both_exist = bool(gpt_exists) and bool(gc_exists)
        both_reject = (gpt_exists is False) and (gc_exists is False)
        disagreement = (gpt_exists is not None and gc_exists is not None
                        and gpt_exists != gc_exists)

        if both_exist:
            if gpt_connotation and gc_connotation and gpt_connotation == gc_connotation:
                score = 55
                reasons.append(
                    "обе модели согласны и коннотация совпадает, НО нет "
                    "независимого подтверждения словарём — согласие двух LLM "
                    "само по себе слабый сигнал (см. задокументированные "
                    "синхронные ошибки AND на словах вроде 'домечек', 'сосенища')"
                )
            else:
                score = 45
                reasons.append(
                    "обе модели согласны, что форма существует, но коннотация "
                    "расходится, и нет подтверждения словарём"
                )
        elif disagreement:
            score = 45
            reasons.append("модели разошлись во мнении о существовании формы — требует проверки")
        elif both_reject:
            score = 10
            reasons.append("обе модели отклонили форму")
        else:
            score = 20
            reasons.append("недостаточно данных от LLM (сбой ответа одной или обеих моделей)")

    # Постфильтр — детерминированное правило перебивает LLM в любую
    # сторону вниз: если чередование обязательно и отсутствует, это
    # системная лингвистическая ошибка, а не вопрос мнения моделей.
    if postfilter_rejected:
        score = min(score, 15)
        reasons.append("нарушено обязательное чередование/вставка гласной (детерминированное правило)")

    bonus = _productivity_bonus(suffix_productivity_note)
    if bonus:
        score = max(0, min(100, score + bonus))
        reasons.append(f"поправка на продуктивность суффикса: {bonus:+d}")

    if score >= 70:
        label = "высокая"
    elif score >= 40:
        label = "средняя"
    else:
        label = "низкая"

    return {"score": score, "label": label, "reasons": reasons}


def differs_only_by_soft_sign(a: str, b: str) -> bool:
    """
    True, если строки отличаются РОВНО одной вставленной/пропущенной
    буквой "ь" (свинка/свинька, свиненка/свиненька) — при прочих равных
    маловероятно, что оба орфографических варианта реальны одновременно
    для одной и той же лексемы: мягкий знак либо нужен, либо нет, это
    не два разных суффикса, а один и тот же с расхождением написания.

    Общий алгоритм по строкам вместо списка суффиксных пар: раньше
    (ошибочно) пары задавались по именам суффиксных тегов (напр.
    "онька"/"енька"), что перепутало разные по смыслу суффиксы. Сравнение
    самих строк не подвержено этой ошибке и не требует список вообще.
    """
    if abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    for i, ch in enumerate(longer):
        if ch == "ь" and longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def resolve_suffix_pair_conflicts(accepted_forms: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Принимает список ПРИНЯТЫХ (exists=True) форм ОДНОЙ леммы. Возвращает
    (оставшиеся_в_основном_списке, перенесённые_на_проверку).

    Если пара (различие ровно в "ь") подтверждена словарём с ОДНОЙ
    стороны — словарно подтверждённая форма остаётся в основном списке
    без изменений, а НЕподтверждённая переносится в needs_review (не
    просто помечается предупреждением): раз словарь уже выбрал более
    вероятный вариант написания, оставлять рядом неподтверждённого
    "конкурента" в основном ответе смысла нет — так основной список
    остаётся набором точных форм, а не форм-с-оговорками.

    Если словарь молчит по ОБЕИМ — обе переносятся в needs_review
    (как и раньше): решение реально неоднозначно, автоматика не должна
    выбирать за эксперта.
    """
    to_move: set[str] = set()
    n = len(accepted_forms)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = accepted_forms[i], accepted_forms[j]
            if not differs_only_by_soft_sign(r1["form"], r2["form"]):
                continue

            dict1, dict2 = r1.get("in_opencorpora", False), r2.get("in_opencorpora", False)
            if dict1 and not dict2:
                r2["pair_conflict"] = (
                    f"отклонено в пользу '{r1['form']}' (отличается только "
                    f"мягким знаком) — та форма подтверждена словарём, эта — нет"
                )
                to_move.add(r2["form"])
            elif dict2 and not dict1:
                r1["pair_conflict"] = (
                    f"отклонено в пользу '{r2['form']}' (отличается только "
                    f"мягким знаком) — та форма подтверждена словарём, эта — нет"
                )
                to_move.add(r1["form"])
            elif not dict1 and not dict2:
                note = (
                    "конфликт написания через мягкий знак с '{other}' — "
                    "маловероятно, что оба варианта реальны одновременно, "
                    "словарь не подтверждает ни один"
                )
                r1["pair_conflict"] = note.format(other=r2["form"])
                r2["pair_conflict"] = note.format(other=r1["form"])
                to_move.add(r1["form"])
                to_move.add(r2["form"])
            # оба в словаре — оба легитимны, ничего не переносим

    moved = [r for r in accepted_forms if r["form"] in to_move]
    remaining = [r for r in accepted_forms if r["form"] not in to_move]
    return remaining, moved


def classify_for_pipeline(final_exists: bool, confidence: dict) -> str:
    """
    КРИТИЧЕСКИ ВАЖНО: score НЕ меняет то, что попадает в основной вывод
    пользователю. final_exists — это уже принятое пайплайном решение
    (AND + постфильтр + словарь, БЕЗ изменений). Score используется
    только для того, чтобы разложить результат по ТРЁМ раздельным
    каналам, а не для того, чтобы задним числом включить в основной
    список формы, которые AND отклонил.

    Возвращает одну из трёх меток:
        "accepted"      — идёт в обычный вывод пользователю, как сейчас.
        "needs_review"  — НЕ идёт пользователю; отдельная очередь для
                           будущей экспертной верификации лингвистом
                           (напр. разногласие LLM: одна модель "да",
                           другая "нет" — сейчас такие формы просто
                           теряются, здесь они видны эксперту отдельно,
                           но не смешиваются с подтверждёнными).
        "rejected"      — низкий score, ни туда ни туда.

    Форма с disagreement НИКОГДА не получает "accepted" через этот
    механизм, даже если у неё "средний" score — попадание в основной
    список определяется ИСКЛЮЧИТЕЛЬНО final_exists, как и раньше.
    """
    if final_exists:
        return "accepted"
    if confidence["label"] == "средняя":
        return "needs_review"
    return "rejected"


if __name__ == "__main__":
    cases = [
        # (описание, gpt_exists, gc_exists, gpt_conn, gc_conn, in_dict, postfilter_rejected)
        ("оба согласны + словарь", True, True, "ласкательное", "ласкательное", True, False),
        ("оба согласны, коннотация расходится", True, True, "ласкательное", "нейтральное", False, False),
        ("разногласие моделей (сейчас теряется под AND)", True, False, "ласкательное", None, False, False),
        ("оба согласны, но нарушено чередование (рукочка)", True, True, "ласкательное", "ласкательное", False, True),
        ("оба отклонили", False, False, None, None, False, False),
    ]
    for desc, ge, gce, gconn, gcconn, in_dict, pf in cases:
        result = compute_confidence(ge, gce, gconn, gcconn, in_dict, pf)
        print(f"{desc}:\n  {result}\n")