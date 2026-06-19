from flask import Flask, render_template, request
import re
import json
from collections import defaultdict

import pymorphy2
from docx import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# ======================================================
# GPT
# ======================================================


load_dotenv()
course_api_key = os.getenv("course_api_key")
llm = ChatOpenAI(
    api_key=course_api_key,
    temperature=0.2,
    model="gpt-4o-mini",
    base_url="https://api.vsellm.ru/",
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)

# ======================================================
# pymorphy2
# ======================================================

morph = pymorphy2.MorphAnalyzer()

# ======================================================
# загрузка документа школы
# ======================================================

def load_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text


TEXT = load_docx("school_research.docx")

# ======================================================
# статистика суффиксов из исследования
# ======================================================

SUFFIX_WEIGHTS = {
    "-ик-": 802,
    "-чик-": 167,
    "-очк-": 165,
    "-ечк-": 165,
    "-ушк-": 147,
    "-ишк-": 147,
    "-ешк-": 147,
    "-ок-": 121,
    "-к-": 116
}

# ======================================================
# примеры из документа
# ======================================================

EXAMPLES = """
театр -> театрик, театришко, театрюшка
актриса -> актрисочка, актрисонька, актрисушка, актрисулька, актриска
голос -> голосок, голосочек, голосишко
сюжет -> сюжетик
"""

# ======================================================
# извлечение форм из документа школы
# ======================================================

def extract_forms_from_document(word):
    word = word.lower()
    results = []
    lines = TEXT.split("\n")
    found_section = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() == word:
            found_section = True
            continue
        if found_section:
            if re.fullmatch(r"[А-ЯЁA-Z][а-яёa-z]+", line):
                break
            matches = re.findall(r"[А-ЯЁа-яё]+", line)
            for m in matches:
                m = m.lower()
                if len(m) > 4:
                    results.append(m)
    return list(set(results))

# ======================================================
# проверка словарности
# ======================================================

def is_dictionary_word(word):
    parses = morph.parse(word)
    for p in parses:
        if "DictionaryAnalyzer" in str(p.methods_stack):
            return True
    return False

# ======================================================
# определение суффикса
# ======================================================

KNOWN_SUFFIXES = [
    "очечка",
    "ушечка",
    "онька",
    "енька",
    "очка",
    "ечка",
    "ушка",
    "юшка",
    "ишка",
    "улька",
    "чик",
    "ик",
    "ок",
    "ек",
    "ка"
]

def detect_suffix(word):
    for suffix in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix):
            return suffix
    return ""

# ======================================================
# Модель
# ======================================================

def generate_candidates(word):
    prompt = f"""
    Ты — специалист по русской морфологии и диминутивам.

    Контекст:
    - Диминутивы образуются с помощью суффиксов:
      -ик-, -чик-, -очк-, -ечк-, -ушк-, -ишк-, -ешк-, -ульк-.
    - Возможны:
      1) уменьшительные,
      2) ласкательные,
      3) пренебрежительные формы.
    - Не все теоретически возможные формы существуют.
    - Не выдумывай слова.
    - Если форма окказиональная или искусственная,
      НЕ включай её.
    - Пример:
      досочка — существует,
      досечка — не существует.

    Примеры диминутивов из исследования:

    {EXAMPLES}

    Задача:
    Для слова "{word}"
    Найди только реально существующие диминутивы русского языка.

    Верни JSON:

    {{
      "word":"{word}",
      "diminutives":[
        {{
          "form":"...",
          "style":"..."
        }}
      ]
    }}
    """

    response = llm.invoke(prompt)
    return json.loads(response.content)

# ======================================================
# рейтинг
# ======================================================

def score_form(form, source_school, source_gpt):
    score = 0
    if source_school:
        score += 60
    if source_gpt:
        score += 20
    if is_dictionary_word(form):
        score += 20
    suffix = detect_suffix(form)
    if suffix:
        key = f"-{suffix}-"
        if key in SUFFIX_WEIGHTS:
            score += min(SUFFIX_WEIGHTS[key] // 100, 10)
    return score

# ======================================================
# основной пайплайн
# ======================================================

def build_entry(word):
    merged = defaultdict(
        lambda: {
            "school": False,
            "gpt": False,
            "styles": set()
        }
    )
    # ------------------------------------
    # формы из документа школы
    # ------------------------------------
    school_forms = extract_forms_from_document(
        word
    )
    for form in school_forms:
        merged[form]["school"] = True
    # ------------------------------------
    # GPT
    # ------------------------------------
    try:
        data = generate_candidates(word)
        for item in data["diminutives"]:
            form = item["form"].lower()
            merged[form]["gpt"] = True
            merged[form]["styles"].add(item.get("style", ""))
    except Exception as e:
        print(e)
    # ------------------------------------
    # результат
    # ------------------------------------
    result = []
    for form, info in merged.items():
        score = score_form(
            form,
            info["school"],
            info["gpt"]
        )
        result.append({
            "form": form,
            "suffix": detect_suffix(form),
            "dictionary_word": is_dictionary_word(form),
            "source_gpt": info["gpt"],
            "score": score,
            "styles": list(info["styles"])
        })
    result.sort(key=lambda x: x["score"], reverse=True)
    return {
        "lemma": word,
        "diminutives": result
    }

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    word = ""

    if request.method == "POST":

        word = request.form.get("word", "").strip()

        if word:
            result = build_entry(word)

    return render_template(
        "index.html",
        result=result,
        word=word
    )

if __name__ == "__main__":
    app.run(debug=True)