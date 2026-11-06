"""Предобработка текста отзыва: очистка HTML-разметки."""
import html
import re


def clean_html(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()