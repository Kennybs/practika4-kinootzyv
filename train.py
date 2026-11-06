"""Обучение модели тональности отзывов («Киноотзыв», IMDB) и упаковка артефактов."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.preprocessing import clean_html

КОРЕНЬ = Path(__file__).resolve().parent
ДАННЫЕ = КОРЕНЬ / "data" / "IMDB Dataset.csv"
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"

ЦЕЛЬ = "sentiment"
ПОРОГ = 0.60
ЦЕЛЕВОЙ_F1 = 0.85
ВЕРСИЯ_МОДЕЛИ = "kinootzyv-tfidf-logreg-1.1.0"
ЗЕРНО = 42


def метрики(факт, вероятность, порог=ПОРОГ):
    прогноз = (вероятность >= порог).astype(int)
    return {
        "f1_macro": round(float(f1_score(факт, прогноз, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(факт, прогноз)), 4),
        "precision_positive": round(float(precision_score(факт, прогноз, zero_division=0)), 4),
        "recall_positive": round(float(recall_score(факт, прогноз, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(факт, вероятность)), 4),
        "порог": порог,
    }


def main():
    if not ДАННЫЕ.exists():
        raise SystemExit(f"нет файла данных: {ДАННЫЕ}\nположите IMDB Dataset.csv в папку data")
    таблица = pd.read_csv(ДАННЫЕ)
    if set(таблица.columns) != {"review", ЦЕЛЬ} or len(таблица) != 50_000:
        raise ValueError(f"Неожиданная структура: {таблица.shape}, {list(таблица.columns)}")
    print(f"данные: {ДАННЫЕ.name}, строк {len(таблица)}")

    таблица[ЦЕЛЬ] = таблица[ЦЕЛЬ].astype(str).str.strip().str.lower()
    таблица = таблица.drop_duplicates(subset=["review"]).reset_index(drop=True)
    таблица["цель"] = таблица[ЦЕЛЬ].map({"negative": 0, "positive": 1}).astype(int)
    print(f"после удаления дублей: {len(таблица)}")

    X_обуч, X_вал, y_обуч, y_вал = train_test_split(
        таблица["review"], таблица["цель"],
        test_size=0.2, random_state=ЗЕРНО, stratify=таблица["цель"])

    модель = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_html, ngram_range=(1, 2),
                                  max_features=100_000, min_df=2, sublinear_tf=True)),
        ("logreg", LogisticRegression(C=2.0, max_iter=1000,
                                      solver="liblinear", random_state=ЗЕРНО)),
    ])
    модель.fit(X_обуч, y_обуч)
    вероятность = модель.predict_proba(X_вал)[:, 1]
    мет = метрики(y_вал, вероятность)
    print("метрики на валидации:", мет)
    if мет["f1_macro"] < ЦЕЛЕВОЙ_F1:
        raise SystemExit(f"F1-macro {мет['f1_macro']} ниже целевого {ЦЕЛЕВОЙ_F1}")

    АРТЕФАКТЫ.mkdir(parents=True, exist_ok=True)
    joblib.dump(модель, АРТЕФАКТЫ / "model.joblib", compress=3)
    отзывы_вал = X_вал.reset_index(drop=True)
    эталон = []
    for i in range(5):
        p = float(вероятность[i])
        эталон.append({
            "review": отзывы_вал.iloc[i],
            "sentiment": int(p >= ПОРОГ),
            "probability": round(p, 4),
        })
    (АРТЕФАКТЫ / "etalon.json").write_text(
        json.dumps(эталон, ensure_ascii=False, indent=2), encoding="utf-8")
    (АРТЕФАКТЫ / "meta.json").write_text(json.dumps({
        "версия_модели": ВЕРСИЯ_МОДЕЛИ,
        "версия_данных": f"{ДАННЫЕ.name}:{len(таблица)}",
        "обучена": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "признаки": ["review"],
        "цель": ЦЕЛЬ,
        "порог": ПОРОГ,
        "метрики": мет,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("артефакты:", sorted(p.name for p in АРТЕФАКТЫ.iterdir()))


if __name__ == "__main__":
    main()