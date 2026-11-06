"""Схемы запроса и ответа: контракт сервиса."""
from typing import Optional
from pydantic import BaseModel, Field


class Отзыв(BaseModel):
    review: str = Field(..., min_length=1, max_length=20_000,
                        description="текст отзыва (английский)")


class Прогноз(BaseModel):
    sentiment: int = Field(..., description="1 — positive, 0 — negative")
    probability: float = Field(..., description="вероятность класса positive")
    threshold: float = Field(..., description="порог отсечения")
    model_version: str = Field(..., description="версия модели")


class Здоровье(BaseModel):
    status: str
    model_version: str
    model_loaded: bool


class СведенияОМодели(BaseModel):
    model_version: str
    data_version: str
    trained_at: str
    features: list[str]
    target: str
    threshold: float
    metrics: dict


class Ошибка(BaseModel):
    detail: str
    code: Optional[str] = None