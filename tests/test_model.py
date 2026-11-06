"""Тесты модели и артефактов: эталонные предсказания и метаданные."""
import json
import unittest
from pathlib import Path

import joblib

АРТЕФАКТЫ = Path(__file__).resolve().parents[1] / "artifacts"


class ТестАртефактов(unittest.TestCase):
    def test_артефакты_существуют(self):
        for имя in ("model.joblib", "meta.json", "etalon.json"):
            self.assertTrue((АРТЕФАКТЫ / имя).exists(), f"нет артефакта {имя}")

    def test_метаданные_полные(self):
        мета = json.loads((АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
        for ключ in ("версия_модели", "версия_данных", "метрики", "порог"):
            self.assertIn(ключ, мета)

    def test_метрики_не_ниже_целевых(self):
        мета = json.loads((АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(мета["метрики"]["f1_macro"], 0.85)

    def test_эталонные_предсказания(self):
        модель = joblib.load(АРТЕФАКТЫ / "model.joblib")
        мета = json.loads((АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
        эталон = json.loads((АРТЕФАКТЫ / "etalon.json").read_text(encoding="utf-8"))
        self.assertEqual(len(эталон), 5)
        порог = float(мета["порог"])
        for пример in эталон:
            вероятность = float(модель.predict_proba([пример["review"]])[:, 1][0])
            self.assertAlmostEqual(вероятность, пример["probability"], delta=0.01)
            self.assertEqual(int(вероятность >= порог), пример["sentiment"])


if __name__ == "__main__":
    unittest.main()