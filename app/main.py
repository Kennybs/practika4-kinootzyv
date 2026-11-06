"""Сервис тональности отзывов «Киноотзыв» на FastAPI."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import Здоровье, Ошибка, Прогноз, Отзыв, СведенияОМодели

КОРЕНЬ = Path(__file__).resolve().parent.parent
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"
СОСТОЯНИЕ = {}


def загрузить_артефакты():
    try:
        СОСТОЯНИЕ["модель"] = joblib.load(АРТЕФАКТЫ / "model.joblib")
        СОСТОЯНИЕ["мета"] = json.loads(
            (АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        СОСТОЯНИЕ["модель"] = None
        СОСТОЯНИЕ["мета"] = {}


загрузить_артефакты()


@asynccontextmanager
async def жизненный_цикл(app: FastAPI):
    загрузить_артефакты()
    yield


app = FastAPI(title="Сервис тональности отзывов «Киноотзыв»",
              version=(СОСТОЯНИЕ.get("мета") or {}).get("версия_модели", "unknown"),
              lifespan=жизненный_цикл)


@app.get("/health", response_model=Здоровье, tags=["служебные"])
def health():
    мета = СОСТОЯНИЕ.get("мета") or {}
    return Здоровье(
        status="ok" if СОСТОЯНИЕ.get("модель") is not None else "degraded",
        model_version=мета.get("версия_модели", "unknown"),
        model_loaded=СОСТОЯНИЕ.get("модель") is not None,
    )


@app.post("/predict", response_model=Прогноз, tags=["прогноз"],
          responses={422: {"model": Ошибка}, 503: {"model": Ошибка}})
def predict(отзыв: Отзыв):
    модель = СОСТОЯНИЕ.get("модель")
    if модель is None:
        raise HTTPException(status_code=503, detail="модель не загружена")
    мета = СОСТОЯНИЕ.get("мета") or {}
    порог = float(мета.get("порог", 0.5))
    вероятность = float(модель.predict_proba([отзыв.review])[:, 1][0])
    return Прогноз(
        sentiment=int(вероятность >= порог),
        probability=round(вероятность, 4),
        threshold=порог,
        model_version=мета.get("версия_модели", "unknown"),
    )


@app.get("/model-info", response_model=СведенияОМодели, tags=["служебные"])
def model_info():
    мета = СОСТОЯНИЕ.get("мета")
    if not мета:
        raise HTTPException(status_code=503, detail="метаданные недоступны")
    return СведенияОМодели(
        model_version=мета.get("версия_модели", "unknown"),
        data_version=мета.get("версия_данных", "unknown"),
        trained_at=мета.get("обучена", "unknown"),
        features=мета.get("признаки", []),
        target=мета.get("цель", "sentiment"),
        threshold=float(мета.get("порог", 0.5)),
        metrics=мета.get("метрики", {}),
    )