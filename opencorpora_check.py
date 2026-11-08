"""
    - по OpenCorpora форма НЕ восстанавливается: словарь подтверждает
      существование слова, но не его связь с леммой. Флаг in_opencorpora
      учитывается в confidence и выводит форму на экспертную проверку.

ВАЖНО, почему НЕ нужен HTTP/API:
    У самого OpenCorpora нет публичного API для проверки одного слова —
    только загрузка полного XML-корпуса (сотни МБ/несколько ГБ) и
    веб-интерфейс для разметки текстов. pymorphy2 уже содержит
    скомпилированный словарь OpenCorpora локально (устанавливается через
    pip install pymorphy2-dicts-ru), поэтому проверка мгновенная и
    полностью бесплатная — вызывать внешний сервис не нужно и не за чем.

ВАЖНО, что это ПРОВЕРЯЕТ и чего НЕ проверяет:
    - word_is_known(word) отвечает True, если слово (в любой форме) есть
      в словаре OpenCorpora. Это касается и словарных, литературных слов,
      и просторечных, но реально распространённых форм — авторы pymorphy2
      явно предупреждают, что словарь "содержит и некорректные формы,
      которые часто употребляются".
    - Это НЕ гарантия того, что слова НЕТ, если word_is_known вернул
      False — редкие разговорные диминутивы вполне может не быть в
      корпусе, даже если носители языка их используют (корпус строится
      на опубликованных текстах, а не на живой разговорной речи/опросах).
    - Поэтому: True от word_is_known — сильный сигнал "слово точно
      существует" (можно доверять больше, чем ответу LLM). False — слабый
      сигнал "не найдено в этом конкретном корпусе", не более того.
"""

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

import json
import os

ATTESTED_PATH = "attested_forms.json"


def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def _load_attested(path: str = ATTESTED_PATH) -> set[tuple[str, str]]:
    """Пары (лемма, форма) из словаря суффиксов. Нет файла — сигнал просто выключен."""
    if not os.path.exists(path):
        print(f"[opencorpora_check] {path} не найден — проверка по словарю суффиксов отключена")
        return set()
    with open(path, encoding="utf-8") as f:
        return {(_norm(p["lemma"]), _norm(p["form"])) for p in json.load(f)}


ATTESTED_PAIRS = _load_attested()


def _noun_lemma(word: str) -> str:
    """Нормальная форма с приоритетом существительного (как get_noun_parse в generator)."""
    parses = morph.parse(word)
    nouns = [p for p in parses if "NOUN" in p.tag]
    return (max(nouns, key=lambda p: p.score) if nouns else parses[0]).normal_form

PROPER_NOUN_TAGS = {"Surn", "Name", "Patr", "Geox", "Orgn"}


def is_known_noun_lemma(word: str) -> bool:
    """
    Строже, чем word_is_known: форма должна быть САМОСТОЯТЕЛЬНЫМ
    существительным в начальной форме, а не падежной формой другого
    слова ("конок" = род.п. мн.ч. от "конка") и не фамилией/именем.
    Кандидаты генератора всегда в им.п. ед.ч., так что для настоящих
    диминутивов normal_form совпадает с самой формой.
    """
    target = _norm(word)
    for p in morph.parse(word):
        if not p.is_known or "NOUN" not in p.tag:
            continue
        if PROPER_NOUN_TAGS & set(p.tag.grammemes):
            continue
        if _norm(p.normal_form) == target:
            return True
    return False

def is_attested(lemma: str, form: str) -> bool:
    return (_norm(lemma), _norm(form)) in ATTESTED_PAIRS

def is_known_in_opencorpora(word: str) -> bool:
    """
    Самодостаточная проверка — не зависит от внутреннего устройства
    generator.py (раньше был импорт оттуда, но это оказалось хрупко:
    если ваш локальный generator.py устроен иначе — например, считает
    in_opencorpora прямо внутри gen_candidates без отдельной функции —
    импорт падает с ImportError). Здесь всё самостоятельно, работает
    независимо от того, как именно у вас организован generator.py.

    Имя параметра для нестрогого сравнения е/ё отличается между версиями/
    форками библиотеки: в оригинальном pymorphy2 — "strict_ee", в
    pymorphy3 и некоторых сборках — просто "strict", в некоторых вообще
    отсутствует. Пробуем все варианты по очереди.
    """
    try:
        return morph.word_is_known(word, strict_ee=False)
    except TypeError:
        pass
    try:
        return morph.word_is_known(word, strict=False)
    except TypeError:
        return morph.word_is_known(word)


def cross_check_with_dictionary(lemma: str, llm_results: list[dict]) -> list[dict]:
    """
    Принимает вывод LLM-фильтра (после merge_and) и добавляет к каждой форме
    поля "in_opencorpora" и "in_attested" — дополнительные, но не решающие
    сигналы. Единственная автоматическая коррекция — восстановление формы,
    которую LLM отклонила (exists=False):

    - по словарю суффиксов (attested_forms.json): всегда — пара
      «лемма + форма» засвидетельствована в академическом словаре,
      омографы исключены привязкой к лемме;
    - по OpenCorpora: только если хотя бы одна модель форму приняла.
      OpenCorpora проверяет лишь существование строки и не отличает
      омограф (конь -> "конец") от диминутива; если обе модели отклонили
      форму, их вердикт важнее.

    Источник восстановления пишется в "override_source".
    Обратное (LLM приняла, а словари не знают) не трогаем: отсутствие
    в словаре не доказывает, что слова нет.
    """
    result = []
    lemma_norm = _noun_lemma(lemma)
    for r in llm_results:
        known = is_known_noun_lemma(r["form"])
        attested = is_attested(lemma_norm, r["form"])
        entry = dict(r)
        entry["in_opencorpora"] = known
        entry["in_attested"] = attested
        # Автоматически восстанавливаем ТОЛЬКО по словарю суффиксов: он
        # подтверждает пару «лемма -> форма». OpenCorpora подтверждает лишь
        # существование слова, но не связь с леммой (конь -> "кончик",
        # "коник" — реальные слова), поэтому in_opencorpora идёт только
        # в confidence: такие формы попадают в needs_review, а не в принятые.
        if r["exists"] is False and attested:
            entry["exists"] = True
            entry["dictionary_override"] = True
            entry["override_source"] = "attested"
        else:
            entry["dictionary_override"] = False

        result.append(entry)
    return result


if __name__ == "__main__":
    # быстрая проверка на конкретных примерах мусора/пропусков,
    # найденных в реальном dictionary_free.json
    test_words = [
        "ручка", "ручонка", "дельце", "словцо", "времечко",  # должны быть известны
        "разенок", "человеченок", "делцо", "жизнонька",       # вероятно неизвестны (мусор)
    ]
    for w in test_words:
        print(f"{w:<15} in_opencorpora={is_known_in_opencorpora(w)}")