"""
Извлечение частотного списка существительных из HTML-страницы
частотного словаря dict.ruslang.ru (Ляшевская, Шаров).

Файл сохранён пользователем локально (кодировка windows-1251),
поэтому читаем с явным указанием кодировки — иначе кириллица бьётся.
"""

import json
import re
from bs4 import BeautifulSoup


def parse_frequency_list(html_path: str, encoding: str = "windows-1251") -> list[dict]:
    with open(html_path, encoding=encoding) as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    entries = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        rank_text = cells[0].get_text(strip=True)
        lemma = cells[1].get_text(strip=True)
        freq_text = cells[2].get_text(strip=True)
        rank2_text = cells[3].get_text(strip=True)

        # строка-разделитель (rowspan) и заголовок таблицы не содержат
        # числового ранга в первой ячейке — пропускаем их, а не падаем
        if not re.match(r"^\d+$", rank_text):
            continue
        if not lemma or not re.match(r"^[а-яё]+$", lemma, re.IGNORECASE):
            continue  # на всякий случай пропускаем нечисловой мусор/латиницу (напр. 'TRUE' в исходных данных)

        try:
            freq_ipm = float(freq_text)
        except ValueError:
            continue

        entries.append({
            "rank": int(rank_text),
            "lemma": lemma,
            "freq_ipm": freq_ipm,
            "rank_by_freq": int(rank2_text) if rank2_text.isdigit() else None,
        })

    return entries


if __name__ == "__main__":
    import sys

    # В Jupyter/IPython sys.argv содержит служебные аргументы ядра
    # (обычно ['-f', '/путь/к/kernel.json']) — попытка отфильтровать только
    # флаги вида '-f' недостаточна, т.к. значение после флага всё равно
    # похоже на "обычный" аргумент. Поэтому просто определяем, что мы
    # внутри Jupyter, и в этом случае полностью игнорируем sys.argv,
    # используя дефолтные пути (их можно переопределить, вызвав функцию
    # parse_frequency_list(...) напрямую в ячейке).
    running_in_jupyter = "ipykernel" in sys.modules

    if running_in_jupyter:
        html_path = "Новый_частотный_словарь_русской_лексики.html"
        out_path = "frequency_nouns.json"
    else:
        html_path = sys.argv[1] if len(sys.argv) > 1 else "Новый_частотный_словарь_русской_лексики.html"
        out_path = sys.argv[2] if len(sys.argv) > 2 else "frequency_nouns.json"

    entries = parse_frequency_list(html_path)

    print(f"Извлечено лемм: {len(entries)}")
    print("Первые 10:")
    for e in entries[:10]:
        print(f"  {e['rank']:>4}  {e['lemma']:<15} ipm={e['freq_ipm']}")
    print("Последние 5:")
    for e in entries[-5:]:
        print(f"  {e['rank']:>4}  {e['lemma']:<15} ipm={e['freq_ipm']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено в {out_path}")