import json
import os
import re
import time
import uuid
import warnings
import requests
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from generator import (
    get_noun_parse, analyze_stem, is_in_dictionary,
    palatalize, get_fleeting_vowel_stem, EXCEPTIONS,
    gen_neut, resolve_connotation,
)

load_dotenv()
# Подавляем SSL-предупреждения
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# 1. НАСТРОЙКИ GIGACHAT
# ============================================================

GIGACHAT_API_KEY = "ВАШ_КЛЮЧ_ЗДЕСЬ"  # Замените на ваш ключ
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Глобальные переменные для GigaChat
gigachat_token = None
gigachat_token_expires = 0

# ============================================================
# 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================

word_data = {}
all_examples = []
word_list = []
gigachat_api_key = None
SUFFIX_TABLE_DATA = None


# ============================================================
# 3. ФУНКЦИИ GIGACHAT
# ============================================================

def gigachat_get_token(api_key: str, force_refresh: bool = False) -> Optional[str]:
    """Получение Access Token для GigaChat (как в тесте)"""
    global gigachat_token, gigachat_token_expires

    if not force_refresh and gigachat_token and time.time() < gigachat_token_expires:
        return gigachat_token

    api_key = api_key.strip()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers=headers,
            data="scope=GIGACHAT_API_PERS",
            verify=False,
            timeout=30
        )

        if response.status_code == 200:
            token_data = response.json()
            gigachat_token = token_data["access_token"]
            gigachat_token_expires = time.time() + token_data.get("expires_at", 3600)
            print("✅ Токен GigaChat получен")
            return gigachat_token
        else:
            print(f"⚠️ Ошибка получения токена: {response.status_code}")
            print(f"   Ответ: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"⚠️ Ошибка получения токена: {e}")
        return None


def gigachat_invoke(prompt: str, api_key: str, temperature: float = 0.3) -> Optional[str]:
    token = gigachat_get_token(api_key)
    if not token:
        print("❌ Не удалось получить токен GigaChat")
        return None

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": "Ты — эксперт по русскому словообразованию. Отвечай точно и по делу."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "repetition_penalty": 1.1
    }

    try:
        response = requests.post(
            f"{GIGACHAT_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            verify=False,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при запросе к GigaChat: {e}")
        if hasattr(e, 'response') and e.response:
            if e.response.status_code == 401:
                print("   🔄 Токен истёк, пробую обновить...")
                new_token = gigachat_get_token(api_key, force_refresh=True)
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    try:
                        response = requests.post(
                            f"{GIGACHAT_BASE_URL}/chat/completions",
                            headers=headers,
                            json=payload,
                            verify=False,
                            timeout=60
                        )
                        response.raise_for_status()
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    except Exception:
                        pass
        return None


# ============================================================
# 4. SPECIAL_CASES
# ============================================================

SPECIAL_CASES = {
    "рука": {"уменьшительные": ["ручка"], "ласкательные": ["рученька", "ручушка"], "пренебрежительные": ["ручище"]},
    "нога": {"уменьшительные": ["ножка"], "ласкательные": ["ноженька", "ножушка"], "пренебрежительные": ["ножище"]},
    "голова": {"уменьшительные": ["головка"], "ласкательные": ["головушка", "головонька"],
               "пренебрежительные": ["головища"]},
    "собака": {"уменьшительные": ["собачка"], "ласкательные": ["собаченька", "собачушка"],
               "пренебрежительные": ["собачище"]},
    "пень": {"уменьшительные": ["пенёк"], "ласкательные": ["пенёчек"], "пренебрежительные": ["пнище"]},
    "стол": {"уменьшительные": ["столик"], "ласкательные": ["столик", "столочек"],
             "пренебрежительные": ["столище", "столишко"]},
    "работа": {"уменьшительные": ["работка", "работочка"], "ласкательные": ["работушка", "работонька", "рабоченька"],
               "пренебрежительные": ["работишка", "работище", "работёнка"]},
    "цветок": {"уменьшительные": ["цветочек"], "ласкательные": ["цветочек", "цветик"],
               "пренебрежительные": ["цветишко"]},
    "лошадь": {"уменьшительные": ["лошадка"], "ласкательные": ["лошадочка", "лошадонька", "лошадушка"],
               "пренебрежительные": ["лошадёнка"]},
    # Пример для похожих по структуре слов (карман, стакан, чемодан и
    # т.п. — masc, твёрдая основа, без чередований): диванчик реален и
    # для "уменьшительные", и для "ласкательные" одновременно (как
    # столик выше), но во избежание путаницы держим по одной форме на
    # категорию явно, а не полагаемся на дедуп между категориями.
    "диван": {"уменьшительные": ["диванчик"], "ласкательные": ["диванец"],
              "пренебрежительные": ["диванище", "диванишко"]},
    # Свободная генерация (без special_case) давала галлюцинации:
    # "свинчик" — суффикс -чик мужского рода приклеен к femn-слову;
    # "свинюнька"/"свинюша" не существуют как реальные формы. "свинка"
    # при этом вообще не генерировалась, хотя это самая частотная форма.
    "свинья": {"уменьшительные": ["свинка"], "ласкательные": ["свинюшка", "свинюлечка"],
               "пренебрежительные": ["свинюха"]},
    # Раньше свободная генерация путала основу с похожим по написанию
    # словом "кровь" (кровушка, кровиночка) — см. is_confusable_pair/
    # is_stem_consistent ниже, здесь просто фиксируем надёжные формы.
    "кровать": {"уменьшительные": ["кроватка"], "ласкательные": ["кроваточка", "кроватонька"],
                "пренебрежительные": ["кроватушка", "кроватище"]},
    # "иголка" — лексикализованный диминутив (уже воспринимается как
    # нейтральное базовое слово для многих носителей), поэтому в
    # пренебрежительные ничего не заданo — не выдумываем.
    "игла": {"уменьшительные": ["иголка"], "ласкательные": ["иголочка"],
             "пренебрежительные": []}
}

# ============================================================
# 5. PATTERN_RULES
# ============================================================

PATTERN_RULES = {
    "feminine_ость": {
        "pattern": r".*ость$",
        "types": ["женский род", "оканчивается на -ость"],
        "examples": ["радость", "новость", "молодость", "смелость"],
        "diminutives": {
            "уменьшительные": [],
            "ласкательные": [],
            "пренебрежительные": ["{stem}остёнка"]
        },
        "transform": "remove_last_4",
        "notes": "Убираем -ость (новость → нов), добавляем -остёнка: новостёнка"
    },
    "neuter_ое": {
        "pattern": r".*[ое]$",
        "types": ["средний род", "оканчивается на -о/-е"],
        "examples": ["море", "горе", "поле", "село", "окно", "солнце"],
        "use_generator": True,
        # diminutives/transform ниже больше не используются при
        # use_generator=True (сохранены только как справочные метаданные
        # для старого notes-текста и на случай отладки) — реальная
        # генерация теперь идёт через generator.gen_candidates(), которая
        # уже проверена через OpenCorpora и корректно различает твёрдую/
        # мягкую основу (в отличие от прежних ручных шаблонов "{word}чко"/
        # "{stem}юшко", ошибочно применявшихся ко ВСЕМ словам на -о/-е
        # одинаково — что давало "словочко"/"словюшко" вместо реальных
        # "словечко"/"словцо").
        "diminutives": {
            "уменьшительные": ["{word}чко", "{stem}юшко"],
            "ласкательные": ["{word}чко", "{stem}юшко"],
            "пренебрежительные": ["{stem}ишко"]
        },
        "transform": "remove_last_vowel",
        "notes": "Генерируется через generator.py (проверено через OpenCorpora) — твёрдая и мягкая основа обрабатываются по-разному автоматически."
    }
}


# ============================================================
# 6. ЗАГРУЗКА ДАННЫХ (опрос — тематика "театр/хореография")
# ============================================================

def load_data(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_examples(data: Dict) -> List[Dict]:
    examples = []
    for group in ["2_class", "6_class", "students"]:
        if group not in data.get("data", {}):
            continue
        for base_word, types in data["data"][group].items():
            if group == "2_class" and "diminutives" in types:
                for dim, count in types["diminutives"].items():
                    examples.append({
                        "base_word": base_word,
                        "diminutive": dim,
                        "count": count,
                        "type": "diminutives",
                        "respondent_group": group
                    })
            else:
                for type_name, dims in types.items():
                    if not dims:
                        continue
                    for dim, count in dims.items():
                        examples.append({
                            "base_word": base_word,
                            "diminutive": dim,
                            "count": count,
                            "type": type_name,
                            "respondent_group": group
                        })
    return examples


def group_examples_by_word(examples: List[Dict]) -> Dict:
    grouped = {}
    for ex in examples:
        word = ex["base_word"]
        if word not in grouped:
            grouped[word] = []
        grouped[word].append(ex)
    return grouped


def load_and_prepare_data(json_path: str) -> None:
    global word_data, all_examples, word_list
    raw_data = load_data(json_path)
    all_examples = prepare_examples(raw_data)
    grouped = group_examples_by_word(all_examples)
    word_list = list(grouped.keys())
    word_data = {}
    for word, examples in grouped.items():
        word_data[word] = {"examples": examples}
    print(f"✅ Загружено {len(word_list)} слов, {len(all_examples)} примеров (эксперимент)")


# ============================================================
# 6b. ДОПОЛНИТЕЛЬНЫЕ ИСТОЧНИКИ ДЛЯ RETRIEVAL
# ============================================================
# Эксперимент покрывает только 14 лемм тематики "театр/хореография" —
# этого недостаточно для retrieval по произвольному слову ("диван",
# "дверь" и т.п. там просто отсутствуют). Расширяем базу за счёт уже
# существующих в проекте файлов с реально подтверждёнными формами:
# dictionary_free.json (~1000 частотных лемм, подтверждено OpenCorpora)
# и ground_truth_merged.json (вручную/экспериментально подтверждённые
# формы). Никаких новых источников данных заводить не нужно.

CONNOTATION_TO_TYPE = {
    "ласкательное": "affection",
    "нейтральное": "size",
    "пренебрежительное": "pejorative",
}


def load_dictionary_free_examples(path: str = "dictionary_free.json") -> None:
    global word_data, word_list
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {path} не найден, пропускаю дополнительный источник")
        return

    added = 0
    for lemma, entry in data.items():
        if entry.get("status") != "ok" or not entry.get("forms"):
            continue
        bucket = word_data.setdefault(lemma, {"examples": []})
        examples = bucket["examples"]
        for f in entry["forms"]:
            type_key = CONNOTATION_TO_TYPE.get(f.get("connotation_hint"))
            if not type_key:
                continue
            examples.append({
                "base_word": lemma, "diminutive": f["form"], "count": 1,
                "type": type_key, "respondent_group": "dictionary_free",
            })
        if lemma not in word_list and examples:
            word_list.append(lemma)
        added += 1
    print(f"✅ Добавлено из {path}: {added} лемм")


def load_ground_truth_examples(path: str = "ground_truth_merged.json") -> None:
    global word_data, word_list
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {path} не найден, пропускаю дополнительный источник")
        return

    added = 0
    for lemma, forms in data.items():
        bucket = word_data.setdefault(lemma, {"examples": []})
        examples = bucket["examples"]
        for form, entry in forms.items():
            if not entry.get("exists"):
                continue
            type_key = CONNOTATION_TO_TYPE.get(entry.get("connotation"), "diminutives")
            examples.append({
                "base_word": lemma, "diminutive": form, "count": 1,
                "type": type_key, "respondent_group": "ground_truth",
            })
        if lemma not in word_list and examples:
            word_list.append(lemma)
        added += 1
    print(f"✅ Добавлено из {path}: {added} лемм")


CATEGORY_TO_TYPE = {
    "уменьшительные": "size",
    "ласкательные": "affection",
    "пренебрежительные": "pejorative",
}


def load_special_cases_as_examples() -> None:
    """SPECIAL_CASES — вручную проверенные, надёжные формы. Помимо
    прямого хардкод-обхода для точного слова (см. generate_diminutives),
    они полезны и как few-shot примеры для retrieval по ДРУГИМ похожим
    словам: например, курированные формы "свинья" (свинка/свинюшка/
    свинюлечка/свинюха) помогут при генерации для "семья"/"судья" —
    морфологически похожих femn-слов на -ья, которые сами не в
    SPECIAL_CASES и идут через свободную генерацию GigaChat."""
    global word_data, word_list
    added = 0
    for lemma, categories in SPECIAL_CASES.items():
        bucket = word_data.setdefault(lemma, {"examples": []})
        examples = bucket["examples"]
        for category, forms in categories.items():
            type_key = CATEGORY_TO_TYPE.get(category, category)
            for form in forms:
                examples.append({
                    "base_word": lemma, "diminutive": form, "count": 1,
                    "type": type_key, "respondent_group": "special_cases",
                })
        if lemma not in word_list and examples:
            word_list.append(lemma)
        added += 1
    print(f"✅ Добавлено из SPECIAL_CASES как примеры retrieval: {added} лемм")


# ============================================================
# 7. ЗАГРУЗКА ТАБЛИЦЫ СУФФИКСОВ
# ============================================================

def load_suffix_table(filepath: str = "suffix_connotation_table.json") -> List[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Файл {filepath} не найден. Продолжаем без таблицы суффиксов.")
        return []


# ============================================================
# 8. ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# ============================================================
# Без FAISS/sentence-transformers: похожие слова подбираются символьно
# по морфологическим признакам (см. раздел 9), а не по эмбеддингам.
# Причина отказа от эмбеддингов: sentence-transformer модели общего
# назначения (напр. all-MiniLM-L6-v2) обучены на семантическую близость
# ПРЕДЛОЖЕНИЙ, часто в основном на английском корпусе — для отдельных
# русских лемм без контекста они дают шумные векторы, не отражающие
# словообразовательное сходство (реальный пример: "диван" оказывался
# ближе всего к "артист", хотя релевантны совсем другие признаки — род,
# тип основы, чередование согласных).

def initialize_system(json_path: str, api_key: str) -> None:
    global gigachat_api_key, SUFFIX_TABLE_DATA
    gigachat_api_key = api_key

    print("🚀 Инициализация системы (без FAISS, символьный морфологический retrieval)...")

    SUFFIX_TABLE_DATA = load_suffix_table("suffix_connotation_table.json")
    if SUFFIX_TABLE_DATA:
        print(f"✅ Загружена таблица суффиксов: {len(SUFFIX_TABLE_DATA)} записей")
    else:
        print("ℹ️ Таблица суффиксов не загружена")

    token = gigachat_get_token(api_key)
    if token:
        print("✅ GigaChat готов к работе")
    else:
        print("⚠️ Не удалось проверить GigaChat, но система продолжит работу")

    load_and_prepare_data(json_path)   # diminutives_experiment_full.json
    load_dictionary_free_examples()    # dictionary_free.json
    load_ground_truth_examples()       # ground_truth_merged.json
    load_special_cases_as_examples()   # SPECIAL_CASES как few-shot для похожих слов

    print(f"✅ Система готова: {len(word_list)} лемм в базе для retrieval")


# ============================================================
# 9. СИМВОЛЬНЫЙ МОРФОЛОГИЧЕСКИЙ RETRIEVAL (замена FAISS)
# ============================================================
# Признаки — те же, что использует generator.py для принятия решений
# о суффиксах и обязательности чередования: род, тип основы (hard/soft),
# наличие велярного согласного (к/г/х/ц) на конце основы. Похожесть по
# этим признакам структурно релевантна задаче диминутивов, в отличие от
# семантической близости слов.

VELAR = {"к", "г", "х", "ц"}


def get_morph_features(lemma: str) -> Dict:
    parsed = get_noun_parse(lemma)
    gender = parsed.tag.gender
    if gender not in ("masc", "femn", "neut"):
        return {"gender": gender, "kind": None, "velar": False, "stem": None}
    stem, kind = analyze_stem(parsed.normal_form, gender)
    velar = bool(stem) and stem[-1] in VELAR
    return {"gender": gender, "kind": kind, "velar": velar, "stem": stem}


def morph_similarity(f1: Dict, f2: Dict) -> int:
    """Разный род -> не сравниваем вообще (-1, отсекается ниже).
    Совпадение типа основы и наличия велярного чередования весит больше
    всего — это то, что реально меняет набор доступных суффиксов."""
    if f1["gender"] != f2["gender"]:
        return -1
    return (3 if f1["kind"] == f2["kind"] else 0) + (3 if f1["velar"] == f2["velar"] else 0)


def is_confusable_pair(stem1: Optional[str], stem2: Optional[str]) -> bool:
    """True, если одна основа является префиксом другой (напр. "кров"
    для "кровь" — префикс "кроват" для "кровать"). Такие пары проходят
    по морфологическим признакам (тот же род/тип основы/чередование),
    но визуальное сходство основ на практике сбивает LLM с толку — модель
    начинает подмешивать формы ЧУЖОГО слова (реальный случай: few-shot
    "кровь: кровушка/кровиночка" привёл к тому, что эти формы
    сгенерировались для запроса "кровать"). Исключаем такие пары из
    retrieval полностью, а не полагаемся только на пост-фильтр."""
    if not stem1 or not stem2 or stem1 == stem2:
        return False
    return stem1.startswith(stem2) or stem2.startswith(stem1)


def search_similar_words(query: str, top_k: int = 5) -> List[Tuple[str, float, List[Dict]]]:
    """Та же сигнатура, что и у прежней FAISS-версии (слово, score,
    примеры) — build_prompt() и остальной код не меняются.
    min_score=3 отсекает слабые совпадения (только род) — нерелевантный
    пример few-shot хуже, чем его отсутствие."""
    target = get_morph_features(query)
    if target["gender"] not in ("masc", "femn", "neut"):
        return []

    scored = []
    for word in word_list:
        if word == query:
            continue
        cand = get_morph_features(word)
        if is_confusable_pair(target["stem"], cand["stem"]):
            continue
        score = morph_similarity(target, cand)
        if score >= 3:
            scored.append((word, float(score), word_data[word]["examples"]))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ============================================================
# 10. ОПРЕДЕЛЕНИЕ ТИПА И ГЕНЕРАЦИЯ ПО ПАТТЕРНУ
# ============================================================

def get_word_type(word: str) -> Optional[Dict]:
    word_lower = word.lower()
    for rule_name, rule in PATTERN_RULES.items():
        if re.match(rule["pattern"], word_lower):
            return rule
    return None


CONNOTATION_TO_CATEGORY = {
    "нейтральное": "уменьшительные",
    "ласкательное": "ласкательные",
    "пренебрежительное": "пренебрежительные",
}


def generate_via_generator_rules(word: str, max_per_category: int = 3) -> Dict[str, List[str]]:
    """Прямая генерация для слов среднего рода на -о/-е (см.
    PATTERN_RULES["neuter_ое"]), БЕЗ вызова get_noun_parse/gen_candidates
    целиком.

    Почему не gen_candidates(word) напрямую: у некоторых слов есть
    морфологическая омонимия по падежу с другим словом другого рода —
    например, строка "горе" совпадает и с им.п. ср.р. слова "горе"
    (печаль), и с предложным падежом "о горе" слова "гора" (жен.р.).
    pymorphy2 в такой неоднозначности иногда выбирает не тот разбор
    (реально наблюдалось: gen_candidates("горе") сгенерировал формы от
    "гора" — горка/горочка вместо горюшко/горечко от настоящего "горе").
    Раз мы УЖЕ знаем род и тип основы из самого условия срабатывания
    правила (слово среднего рода, оканчивается на о/е) — не нужно
    заново определять их через pymorphy2, работаем со строкой напрямую.

    Фильтр по in_opencorpora строгий (не просто приоритет, а отсев):
    эта ветка работает без LLM-проверки, поэтому, как и в
    build_dictionary_free.py, для неё точность важнее полноты —
    неподтверждённая словарём форма (напр. "словко", "словице") не
    включается, а не просто уходит в конец списка."""
    stem = word[:-1]
    kind = "hard" if word.endswith("о") else "soft"

    raw: set[tuple[str, str]] = set(gen_neut(stem, kind)) | set(EXCEPTIONS.get(word, []))

    # ё/е орфографический дубль — та же логика, что в generator.gen_candidates
    for form, tag in list(raw):
        if "ё" in form:
            raw.add((form.replace("ё", "е"), tag))

    result: Dict[str, List[str]] = {"уменьшительные": [], "ласкательные": [], "пренебрежительные": []}
    seen_forms = set()
    for form, tag in raw:
        if form == word or form in seen_forms:
            continue
        seen_forms.add(form)

        if not is_in_dictionary(form):
            continue  # без LLM-проверки принимаем только словарно подтверждённые

        conn = resolve_connotation(tag)
        category = CONNOTATION_TO_CATEGORY.get(conn["connotation"])
        if not category or len(result[category]) >= max_per_category:
            continue
        if form not in result[category]:
            result[category].append(form)

    return result


def generate_by_pattern(word: str, rule: Dict) -> Dict:
    if rule.get("use_generator"):
        return generate_via_generator_rules(word)

    transform = rule.get("transform", "none")
    if transform == "remove_last":
        stem = word[:-1]
    elif transform == "remove_last_2":
        stem = word[:-2]
    elif transform == "remove_last_3":
        stem = word[:-3]
    elif transform == "remove_last_4":
        stem = word[:-4]
    elif transform == "remove_last_vowel":
        stem = word[:-1]
    else:
        stem = word

    result = {}
    for category, templates in rule["diminutives"].items():
        variants = []
        for t in templates:
            if not t:
                continue
            if "{word}" in t:
                variants.append(t.format(word=word))
            elif "{stem}" in t:
                variants.append(t.format(stem=stem))
        result[category] = variants
    return result


# ============================================================
# 11. ПОСТ-ОБРАБОТКА
# ============================================================
# Две задачи: 1) орфографическая коррекция явных ошибок написания
# (дверька -> дверка), 2) дедупликация НЕ только внутри категории, но
# и МЕЖДУ категориями (прежняя версия могла держать одну и ту же форму
# сразу в двух категориях).

# Правила — общие орфографические паттерны, не хардкод под конкретное
# слово. При необходимости расширять списком (паттерн, замена).
ORTHOGRAPHY_CORRECTIONS = [
    (r"ька$", "ка"),   # дверька -> дверка и подобные твердоосновные случаи
]


def correct_orthography(form: str) -> str:
    """Если форма неизвестна словарю OpenCorpora, а скорректированный
    вариант — известен, берём скорректированный. Если оба варианта
    неизвестны (частый случай для редких разговорных диминутивов) —
    форму не трогаем, чтобы не потерять реальные, но редкие слова."""
    if is_in_dictionary(form):
        return form
    for pattern, replacement in ORTHOGRAPHY_CORRECTIONS:
        corrected = re.sub(pattern, replacement, form)
        if corrected != form and is_in_dictionary(corrected):
            return corrected
    return form


# Известные окончания диминутивных суффиксов мужского рода (см.
# generator.py: gen_masc). Мужской род в им.п. ед.ч. НЕ оканчивается на
# гласные "о"/"а"/"е" (кроме отдельных случаев -ишко, которое само по
# себе валидный пренебрежительный суффикс, поэтому явно в списке) — если
# сгенерированная форма для masc-леммы не оканчивается ни на один из
# известных суффиксов, это, скорее всего, гибрид/галлюцинация вида
# "диванчико" (суффикс -чик + случайно приклеенное чужеродное -о).
KNOWN_MASC_DIM_ENDINGS = [
    "ичек", "ишко", "онок", "ёнок", "чик", "ик", "ок", "ек", "ёк",
    "ец", "очек", "ёчек", "ище",
]


def is_plausible_gendered_form(lemma: str, form: str) -> bool:
    """Проверяет согласованность окончания формы с родом леммы. Сейчас
    правило есть только для masc (самый частый источник таких гибридов
    в свободной генерации LLM); для femn/neut orth-вариантов в
    generator.py слишком много, чтобы дать надёжный белый список без
    риска отбросить реальные формы — там полагаемся на GigaChat и
    последующую словарную проверку. Ошибки определения рода (слово не
    из словаря и т.п.) не должны блокировать результат — fail open."""
    try:
        parsed = get_noun_parse(lemma)
        gender = parsed.tag.gender
    except Exception:
        return True

    if gender != "masc":
        return True

    return any(form.endswith(e) for e in KNOWN_MASC_DIM_ENDINGS)


def get_valid_stem_prefixes(norm_lemma: str, gender: str) -> List[str]:
    """Все допустимые начала формы для леммы: наивная основа, её
    палатализованный вариант, и то же для альтернативной основы с
    беглой гласной (если есть) — та же логика, что использует сам
    generator.py при генерации кандидатов."""
    if gender not in ("masc", "femn", "neut"):
        return []
    stem, kind = analyze_stem(norm_lemma, gender)
    prefixes = {stem, palatalize(stem)}
    alt_stem = get_fleeting_vowel_stem(norm_lemma, gender, stem)
    if alt_stem:
        prefixes.add(alt_stem)
        prefixes.add(palatalize(alt_stem))
    return [p for p in prefixes if p]


def is_stem_consistent(lemma: str, form: str) -> bool:
    """Проверяет, что форма реально образована от основы ИМЕННО этой
    леммы, а не другого слова, похожего по написанию, случайно попавшего
    в few-shot (реальный случай: "кровать" получила формы "кровушка"/
    "кровиночка" — они образованы от основы слова "кровь", а не
    "кровать"). Иррегулярные основы (EXCEPTIONS в generator.py, напр.
    окно->окошко) не проверяем этим способом — там основа меняется
    непредсказуемо, полагаемся на словарную проверку. Ошибки определения
    рода/разбора не должны блокировать результат — fail open."""
    try:
        parsed = get_noun_parse(lemma)
        norm_lemma = parsed.normal_form
        gender = parsed.tag.gender
    except Exception:
        return True

    if norm_lemma in EXCEPTIONS:
        return True

    prefixes = get_valid_stem_prefixes(norm_lemma, gender)
    if not prefixes:
        return True

    return any(form.startswith(p) for p in prefixes)


def clean_diminutives_result(result: Dict, base_word: Optional[str] = None, skip_stem_check: bool = False) -> Dict:
    if "diminutives" not in result:
        return result

    # base_word для проверки рода: явно переданный параметр приоритетнее
    # поля из самого result (которое иногда может отсутствовать/быть
    # искажено, если модель не вернула его в JSON).
    lemma_for_check = base_word or result.get("base_word")

    # Фиксированный порядок категорий -> дедуп детерминирован (при
    # конфликте между категориями форма остаётся в категории с более
    # высоким приоритетом, а не в случайной по порядку словаря).
    category_order = ["уменьшительные", "ласкательные", "пренебрежительные"]
    ordered = [(c, result["diminutives"].get(c, [])) for c in category_order]
    ordered += [(c, v) for c, v in result["diminutives"].items() if c not in category_order]

    seen_global = set()
    cleaned = {}
    for category, values in ordered:
        if not isinstance(values, list):
            cleaned[category] = values
            continue
        unique = []
        for v in values:
            if not v or not v.strip():
                continue
            v = correct_orthography(v.strip())
            # Проверки ниже защищают от галлюцинаций СВОБОДНОЙ генерации
            # GigaChat. Для SPECIAL_CASES/PATTERN_RULES (skip_stem_check=
            # True) их применять не нужно и вредно: это вручную
            # проверенные данные, которые могут законно нарушать наивные
            # эвристики (напр. "иголка" для "игла" — историческое
            # чередование л/лк, не покрытое вычислением наивной основы).
            if not skip_stem_check:
                if lemma_for_check and not is_plausible_gendered_form(lemma_for_check, v):
                    continue  # гибрид вида "диванчико" — суффикс не согласован с родом леммы
                if lemma_for_check and not is_stem_consistent(lemma_for_check, v):
                    continue  # форма образована от основы ДРУГОГО слова (напр. "кровушка" для "кровать")
            if v in seen_global:
                continue
            seen_global.add(v)
            unique.append(v)
        cleaned[category] = unique

    result["diminutives"] = cleaned
    return result


# ============================================================
# 12. ФОРМИРОВАНИЕ ПРОМПТА (с таблицей суффиксов)
# ============================================================

def build_prompt(query_word: str, similar_words: List[Tuple[str, float, List[Dict]]]) -> str:
    global SUFFIX_TABLE_DATA

    prompt = f"""Ты — эксперт по русскому словообразованию. Твоя задача — образовать диминутивы (уменьшительные, ласкательные и пренебрежительные формы) от слова "{query_word}".

Ниже приведены примеры того, как носители русского языка образуют диминутивы от похожих слов (похожих по роду и типу основы). Используй эти примеры как образец для подражания. Отвечай строго на основе этих примеров, не придумывай новых слов!

## 📚 ПРИМЕРЫ ИЗ БАЗЫ ЗНАНИЙ

"""

    for word, distance, examples in similar_words:
        if not examples:
            continue
        prompt += f"### Слово: {word}\n"
        by_type = {}
        for ex in examples:
            type_key = ex["type"]
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(
                f"{ex['diminutive']} (встречается {ex['count']} раз, группа: {ex['respondent_group']})"
            )
        type_labels = {
            "size": "Уменьшительные",
            "affection": "Ласкательные",
            "pejorative": "Пренебрежительные",
            "diminutives": "Уменьшительно-ласкательные"
        }
        for type_key, items in by_type.items():
            label = type_labels.get(type_key, type_key)
            prompt += f"  {label}: {', '.join(items)}\n"
        prompt += "\n"

    # Добавляем информацию о суффиксах из таблицы
    if SUFFIX_TABLE_DATA:
        prompt += "\n## 📌 ДОСТУПНЫЕ СУФФИКСЫ ДЛЯ ДИМИНУТИВОВ\n\n"
        groups = {
            "ласкательное": [],
            "пренебрежительное": [],
            "нейтральное": [],
            "смешанное": []
        }
        for entry in SUFFIX_TABLE_DATA:
            primary = entry.get("connotation_primary", "").lower()
            if primary in groups:
                groups[primary].append(entry)
            else:
                groups["смешанное"].append(entry)

        for connotation, entries in groups.items():
            if not entries:
                continue
            prompt += f"### {connotation.capitalize()} суффиксы:\n"
            for entry in entries:
                suffix = entry.get("suffix", "")
                orth = entry.get("orth_variants", [])
                note = entry.get("note", "")
                if note and len(note) > 80:
                    note = note[:80] + "..."
                prompt += f"- **-{suffix}-** (варианты: {', '.join(orth[:3])}{'...' if len(orth) > 3 else ''}) — {note}\n"
            prompt += "\n"

    prompt += f"""## 🎯 ЗАДАНИЕ

Образуй диминутивы для слова **{query_word}** в трёх категориях:

1. **Уменьшительные** — 0-3 РАЗНЫХ варианта (без повторов!)
2. **Ласкательные** — 0-3 РАЗНЫХ варианта (без повторов!)
3. **Пренебрежительные** — 0-3 РАЗНЫХ варианта (без повторов!)

Если для какой-то категории реально не существует диминутивной формы —
верни для неё пустой список []. НЕ придумывай форму только ради того,
чтобы категория не была пустой.

**ВАЖНО:**
- НЕ ПОВТОРЯЙ один и тот же вариант внутри одной категории и МЕЖДУ категориями!
- Используй суффиксы, которые встречаются в примерах выше, а также из списка доступных суффиксов.
- Для уменьшительных форм предпочтительны суффиксы, помеченные как нейтральные или ласкательные.
- Для ласкательных — суффиксы с пометкой «ласкательное».
- Для пренебрежительных — суффиксы с пометкой «пренебрежительное».
- Не придумывай новые слова!

**ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:**

```json
{{
  "base_word": "{query_word}",
  "diminutives": {{
    "уменьшительные": ["вариант1", "вариант2", ...],
    "ласкательные": ["вариант1", "вариант2", ...],
    "пренебрежительные": ["вариант1", "вариант2", ...]
  }},
  "explanation": "краткое пояснение, какие суффиксы использованы"
}}
```"""
    return prompt


# ============================================================
# 13. ПАРСИНГ ОТВЕТА И ПЕЧАТЬ
# ============================================================

def robust_json_loads(text: str, max_fixes: int = 10) -> Dict:
    """json.loads с авто-починкой самых частых ошибок свободной генерации
    JSON моделью: пропущенная запятая между элементами (наблюдалось на
    практике — "Expecting ',' delimiter") и висячая запятая перед
    закрывающей скобкой. Каждая ошибка чинится по позиции, которую сам
    json.decoder указывает в исключении, и парсинг повторяется — а не
    гадаем вслепую, где именно сломан текст."""
    for _ in range(max_fixes):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if "Expecting ',' delimiter" in e.msg:
                text = text[:e.pos] + "," + text[e.pos:]
                continue
            if ("Expecting property name enclosed in double quotes" in e.msg
                    or "Expecting value" in e.msg):
                # висячая запятая перед '}' (объект) или перед ']' (массив) —
                # два разных сообщения decoder'а для одной и той же ошибки
                prev_comma = text.rfind(",", 0, e.pos)
                if prev_comma != -1:
                    text = text[:prev_comma] + text[prev_comma + 1:]
                    continue
            raise
    raise ValueError(f"Не удалось починить JSON за {max_fixes} попыток")


def extract_diminutives_fallback(text: str) -> Optional[Dict]:
    """Последний рубеж: если JSON не чинится вообще, вытаскиваем списки
    диминутивов напрямую по категориям регулярным выражением, игнорируя
    остальную (возможно поломанную) структуру — нам всё равно нужны
    только сами формы, а не поле "explanation" и т.п."""
    categories = ["уменьшительные", "ласкательные", "пренебрежительные"]
    diminutives = {}
    for cat in categories:
        m = re.search(rf'"{cat}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            diminutives[cat] = re.findall(r'"([^"]*)"', m.group(1))
    if not diminutives:
        return None
    return {"diminutives": diminutives, "explanation": "(восстановлено из повреждённого JSON)"}


def parse_json_from_response(response: str) -> Dict:
    match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    candidate = match.group(1) if match else None
    if candidate is None:
        match = re.search(r'\{.*"base_word".*\}', response, re.DOTALL)
        candidate = match.group(0) if match else response

    try:
        return robust_json_loads(candidate)
    except (json.JSONDecodeError, ValueError):
        fallback = extract_diminutives_fallback(response)
        if fallback:
            print("⚠️ JSON от модели повреждён, использован regex-фолбэк по спискам форм")
            return fallback
        return {"error": "Не удалось извлечь JSON", "raw_response": response}


def print_result(result: Dict):
    print(f"📊 Сгенерированные диминутивы:")
    if "diminutives" in result:
        for dim_type, dims in result["diminutives"].items():
            if dims:
                print(f"  {dim_type}: {', '.join(dims)}")
            else:
                print(f"  {dim_type}: (нет вариантов)")
    if "explanation" in result:
        print(f"\n💡 {result['explanation']}")
    print("\n" + "=" * 60)


# ============================================================
# 14. ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ
# ============================================================

def generate_diminutives(query_word: str, top_k: int = 5) -> Dict:
    global gigachat_api_key

    print(f"\n{'=' * 60}")
    print(f"🔍 ЗАПРОС: {query_word}")
    print('=' * 60)

    # 1. SPECIAL_CASES
    if query_word in SPECIAL_CASES:
        print(f"⭐ Найдено особое слово '{query_word}', использую hardcoded значения")
        result = {
            "base_word": query_word,
            "diminutives": SPECIAL_CASES[query_word],
            "explanation": "Особый случай (чередования в корне или исключение)",
            "query_word": query_word,
            "similar_words_used": [],
            "generated_by": "special_case"
        }
        result = clean_diminutives_result(result, base_word=query_word, skip_stem_check=True)
        print_result(result)
        return result

    # 2. PATTERN_RULES
    rule = get_word_type(query_word)
    if rule:
        print(f"📐 Обнаружен тип слова: {', '.join(rule.get('types', ['неизвестный']))}")
        print(f"   Примеры: {', '.join(rule.get('examples', []))}")
        print(f"   {rule.get('notes', '')}")

        result = {
            "base_word": query_word,
            "diminutives": generate_by_pattern(query_word, rule),
            "explanation": f"Сгенерировано по правилу для {', '.join(rule.get('types', ['']))}: {rule.get('notes', '')}",
            "query_word": query_word,
            "similar_words_used": [],
            "generated_by": "pattern_rule",
            "rule_used": rule.get("types", [])
        }
        result = clean_diminutives_result(result, base_word=query_word, skip_stem_check=True)
        print_result(result)
        return result

    # 3. RETRIEVAL (символьный, без FAISS) + GigaChat
    print("🔎 Слово не найдено в исключениях и паттернах, использую морфологический retrieval + GigaChat...")

    similar = search_similar_words(query_word, top_k)

    print(f"\n📚 Найдено {len(similar)} похожих слов:")
    for word, score, examples in similar:
        print(f"  • {word} (морфологическая близость: {score:.0f})")
        examples_str = ", ".join([
            f"{ex['diminutive']}({ex['type']})"
            for ex in examples[:3]
        ])
        print(f"    Примеры: {examples_str}")

    prompt = build_prompt(query_word, similar)

    print(f"\n🤖 Вызов GigaChat...")
    print("-" * 60)

    response = gigachat_invoke(prompt, gigachat_api_key)

    if response is None:
        print("❌ GigaChat вернул пустой ответ")
        result = {
            "base_word": query_word,
            "error": "GigaChat не ответил",
            "query_word": query_word
        }
        return result

    result = parse_json_from_response(response)
    result = clean_diminutives_result(result, base_word=query_word)
    result["query_word"] = query_word
    result["similar_words_used"] = [word for word, _, _ in similar]
    result["generated_by"] = "retrieval_gigachat"

    print_result(result)
    return result


# ============================================================
# 15. ЗАПУСК (без сохранения в файл)
# ============================================================

def main():
    JSON_PATH = "diminutives_experiment_full.json"
    API_KEY = os.getenv('LLM_GIGA_CHAT_AUTH_KEY')  # Замените на ваш ключ

    initialize_system(JSON_PATH, API_KEY)

    test_words = [
        "слон", "нога", "дело",
        "дом", '', "свинья"
    ]

    for word in test_words:
        result = generate_diminutives(query_word=word, top_k=5)

        # Запись в файл временно отключена
        # output_file = f"result_{word}.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False, indent=2)
        # print(f"📁 Результат сохранён в {output_file}")


if __name__ == "__main__":
    main()
