"""
Детерминированный (не-LLM) постфильтр, отсеивающий два конкретных,
но очень частых класса ошибок:

1. Формы БЕЗ чередования согласных там, где оно обязательно
   (рука -> рукочка вместо ручечка). См. is_missing_required_palatalization.

2. Формы, где чередование применено к основе БЕЗ вставленной беглой
   гласной (палка -> палчка вместо палочка). Слова с беглой гласной
   перед конечным велярным согласным (палка -> палок, ложка -> ложек,
   кошка -> кошек) в реальном языке чередуют согласный только ПОСЛЕ
   вставки гласной: пал-к(а) -> пал-о-к -> пал-о-ч-ка. Генератор же
   может (ошибочно) применить палатализацию прямо к "сырой" основе без
   вставленной гласной — пал-к -> пал-ч -> "палчка", которая является
   орфографически похожей, но не существующей формой. См.
   is_missing_required_vowel_insertion.

Оба правила чередования/вставки гласной уже детерминированно закодированы
в generator.py (используются для ГЕНЕРАЦИИ форм), их же можно использовать
и для ПРОВЕРКИ уже принятых LLM-фильтром форм — без дополнительных
вызовов API.

Это не замена LLM-фильтру (он всё ещё нужен для решения "существует ли
слово вообще" и для коннотации), а дополнительный дешёвый барьер именно
против этих конкретных, уже опознанных системных паттернов ошибок.
"""

import pymorphy2
from generator import analyze_stem, palatalize, get_fleeting_vowel_stem, is_in_dictionary

morph = pymorphy2.MorphAnalyzer()


def is_missing_required_palatalization(lemma: str, form: str) -> bool:
    """
    True, если form построена от НЕпалатализованной основы там, где для
    данной леммы палатализация в принципе применима (основа оканчивается
    на к/г/х/ц) — то есть форма, скорее всего, ошибочна.

    Логика: берём основу (analyze_stem) и её палатализованный вариант
    (palatalize). Если они совпадают — чередование неприменимо к этому
    слову вообще, флагать нечего. Если не совпадают — правильная форма
    почти всегда строится от палатализованной основы; форма, начинающаяся
    с "сырой" (непалатализованной) основы — подозрительна.
    """
    parsed = morph.parse(lemma)[0]
    gender = parsed.tag.gender
    if gender not in ("masc", "femn", "neut"):
        return False

    stem, kind = analyze_stem(parsed.normal_form, gender)
    soft = palatalize(stem)

    if stem == soft:
        return False  # чередование неприменимо к этой основе вообще

    return form.startswith(stem) and not form.startswith(soft)


def is_missing_required_vowel_insertion(lemma: str, form: str) -> bool:
    """
    True, если form построена палатализацией "сырой" основы БЕЗ вставки
    беглой гласной там, где эта гласная обязательна (палка -> палчка
    вместо палочка; ложка -> ложчка вместо ложечка).

    Применимо только к femn/neut (см. ограничение get_fleeting_vowel_stem)
    и только когда beglaya-гласная основа (alt_stem) реально отличается
    от наивной основы. Флагуем формы, начинающиеся с палатализованной
    НАИВНОЙ основы (soft_naive) — это ошибочный вариант, правильный же
    строится от палатализованной АЛЬТЕРНАТИВНОЙ основы с гласной
    (soft_alt). Если soft_naive и soft_alt совпадают — различать нечего,
    флагать не нужно (такое возможно для некоторых типов основ).
    """
    parsed = morph.parse(lemma)[0]
    gender = parsed.tag.gender
    if gender not in ("femn", "neut"):
        return False

    naive_stem, kind = analyze_stem(parsed.normal_form, gender)
    alt_stem = get_fleeting_vowel_stem(parsed.normal_form, gender, naive_stem)
    if alt_stem is None:
        return False  # у этого слова вообще нет беглой гласной — правило неприменимо

    soft_naive = palatalize(naive_stem)
    soft_alt = palatalize(alt_stem)

    if soft_naive == naive_stem:
        return False  # палатализация неприменима к наивной основе вообще

    if soft_naive == soft_alt:
        return False  # оба варианта дают одинаковый результат — различать нечего

    return form.startswith(soft_naive) and not form.startswith(soft_alt)


FIRST_DEGREE_MAP = {"ичек": "ик", "очек": "ок", "ёчек": "ёк"}


def is_missing_first_degree_form(lemma: str, form: str, suffix_tag: str) -> bool:
    """
    True, если form использует суффикс "двойной уменьшительности"
    (ичек/очек/ёчек), а соответствующая форма ПЕРВОЙ степени (ик/ок/ёк
    от той же основы) не подтверждена словарём OpenCorpora.

    Обоснование: -очек-/-ёчек-/-ичек- морфологически — это не
    самостоятельный суффикс, а вторая степень уменьшительности,
    наслаиваемая на уже существующую форму первой степени (пень->пенёк->
    пенёчек; звон->звонок->звоночек; муж->мужик->мужичек). Фонологически
    "домечек" от "дом" построен ничем не хуже, чем "звоночек" от "звон" —
    оба СИНТАКСИЧЕСКИ корректны. Разница в том, что у "звон"/"пен" есть
    реальная, употребимая форма первой степени, а у "дом" её нет
    ("домок"/"домёк" не в ходу) — то есть накладывать вторую степень не
    на что. Проверяем именно это структурное предусловие, а не саму
    двойную форму (которую словарь чаще всего и не знает — она реже
    первой степени, но это не значит, что первая степень тоже отсутствует
    там, где двойная действительно реальна).
    """
    first_degree_ending = FIRST_DEGREE_MAP.get(suffix_tag)
    if not first_degree_ending:
        return False  # правило неприменимо к другим суффиксам

    parsed = morph.parse(lemma)[0]
    gender = parsed.tag.gender
    if gender != "masc":
        return False  # ичек/очек/ёчек в generator.py используются только для masc

    stem, kind = analyze_stem(parsed.normal_form, gender)
    soft = palatalize(stem)
    base = soft if soft != stem else stem

    first_degree_form = base + first_degree_ending
    return not is_in_dictionary(first_degree_form)


# Лексикализованные исключения для -онок/-ёнок у НЕодушевлённых существительных
# (реальные слова, где суффикс закрепился исторически, хотя по общему правилу
# суффикс продуктивен для детёнышей животных). Расширять по мере обнаружения.
ONOK_INANIMATE_EXCEPTIONS = {"бочка", "кадка"}


def is_animacy_mismatch(lemma: str, suffix_tag: str) -> bool:
    """
    True, если суффикс "-онок"/"-ёнок" применён к НЕодушевлённому
    существительному без известного исключения.

    Обоснование: -онок/-ёнок в русском продуктивен почти исключительно
    для детёнышей животных (медвежонок, котёнок) — это грамматическая
    категория, а не просто орфографический паттерн, и pymorphy2 уже даёт
    готовый признак одушевлённости (tag.animacy), не нужно ничего
    выдумывать. Для неодушевлённых предметов ("дом", "стол") эта форма
    в норме не образуется, за исключением узкого списка лексикализованных
    слов (бочонок и т.п.).
    """
    if suffix_tag != "онок":
        return False
    if lemma in ONOK_INANIMATE_EXCEPTIONS:
        return False

    parsed = morph.parse(lemma)[0]
    return parsed.tag.animacy == "inan"


def apply_palatalization_postfilter(lemma: str, llm_results: list[dict]) -> list[dict]:
    """
    Принимает вывод filter_candidates() (после LLM-фильтрации) и
    дополнительно обнуляет exists=True там, где сработало любое из
    четырёх детерминированных правил — независимо от того, что решила
    LLM. Помечает такие записи явным флагом "postfilter_rejected", чтобы
    это было видно в данных, а не выглядело как решение LLM.

    Правила 1-2 (чередования) — фонологические: форма грамматически
    неверна независимо от базового слова. Правила 3-4 (первая степень,
    одушевлённость) — структурные/грамматические: форма сама по себе
    произносима и "похожа" на слово, но нарушает предусловие применения
    суффикса к ЭТОЙ конкретной лемме.
    """
    result = []
    for r in llm_results:
        suffix_tag = r.get("suffix")
        flagged = r["exists"] and (
            is_missing_required_palatalization(lemma, r["form"])
            or is_missing_required_vowel_insertion(lemma, r["form"])
            or is_missing_first_degree_form(lemma, r["form"], suffix_tag)
            or is_animacy_mismatch(lemma, suffix_tag)
        )
        entry = dict(r)
        entry["postfilter_rejected"] = flagged
        if flagged:
            entry["exists"] = False
            entry["connotation"] = None
        result.append(entry)
    return result


if __name__ == "__main__":
    # проверка на примерах именно тех ошибок, что были найдены в тесте
    test_cases_palatalization = [
        ("рука", "рукочка", True, "нет чередования, должно быть ручечка"),
        ("рука", "ручечка", False, "чередование есть, форма верная"),
        ("рука", "рукенька", True, "нет чередования"),
        ("рука", "рученька", False, "чередование есть"),
        ("рука", "рукища", True, "нет чередования, должно быть ручища"),
        ("рука", "ручища", False, "чередование есть"),
        ("актриса", "актриска", False, "чередование неприменимо (нет велярного согласного)"),
        ("дождь", "дождик", False, "чередование неприменимо (д не велярный)"),
    ]

    print("Проверка is_missing_required_palatalization:")
    all_ok = True
    for lemma, form, expected, note in test_cases_palatalization:
        actual = is_missing_required_palatalization(lemma, form)
        status = "OK" if actual == expected else "ОШИБКА"
        if actual != expected:
            all_ok = False
        print(f"  [{status}] {lemma} + {form}: ожидалось={expected}, получено={actual} ({note})")

    test_cases_vowel_insertion = [
        ("палка", "палчка", True, "палатализация без вставки гласной, должно быть палочка"),
        ("палка", "палочка", False, "гласная вставлена и палатализация верная"),
        ("палка", "палченька", True, "тот же баг с другим суффиксом"),
        ("палка", "паленька", False, "вставлена гласная, палатализации тут не требуется (суффикс -еньк-)"),
        ("рука", "ручка", False, "у 'рука' нет беглой гласной (основа 'рук' не имеет кластера согласных)"),
        ("актриса", "актриска", False, "чередование неприменимо, беглой гласной у 'актриса' нет"),
    ]

    print("\nПроверка is_missing_required_vowel_insertion:")
    for lemma, form, expected, note in test_cases_vowel_insertion:
        actual = is_missing_required_vowel_insertion(lemma, form)
        status = "OK" if actual == expected else "ОШИБКА"
        if actual != expected:
            all_ok = False
        print(f"  [{status}] {lemma} + {form}: ожидалось={expected}, получено={actual} ({note})")

    print("\nВСЕ ТЕСТЫ ПРОШЛИ" if all_ok else "\nЕСТЬ РАСХОЖДЕНИЯ — см. выше")