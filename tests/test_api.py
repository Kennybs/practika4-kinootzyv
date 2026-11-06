"""Тесты сервиса: успешный путь, ошибки, служебные эндпоинты."""
import unittest

from fastapi.testclient import TestClient

from app.main import app

КЛИЕНТ = TestClient(app)
ВАЛИДНЫЙ = {"review": "A wonderful little production with brilliant acting and story."}


class ТестЗдоровья(unittest.TestCase):
    def test_health_отвечает(self):
        ответ = КЛИЕНТ.get("/health")
        self.assertEqual(ответ.status_code, 200)
        тело = ответ.json()
        self.assertEqual(тело["status"], "ok")
        self.assertTrue(тело["model_loaded"])


class ТестПрогноза(unittest.TestCase):
    def test_валидный_запрос(self):
        ответ = КЛИЕНТ.post("/predict", json=ВАЛИДНЫЙ)
        self.assertEqual(ответ.status_code, 200)
        тело = ответ.json()
        self.assertIn(тело["sentiment"], (0, 1))
        self.assertGreaterEqual(тело["probability"], 0.0)
        self.assertLessEqual(тело["probability"], 1.0)
        self.assertEqual(тело["threshold"], 0.5)

    def test_негативный_отзыв(self):
        ответ = КЛИЕНТ.post("/predict", json={
            "review": "Terrible movie, boring plot and awful acting, waste of time."})
        self.assertEqual(ответ.status_code, 200)
        self.assertEqual(ответ.json()["sentiment"], 0)

    def test_нет_обязательного_поля(self):
        self.assertEqual(КЛИЕНТ.post("/predict", json={}).status_code, 422)

    def test_пустой_текст(self):
        self.assertEqual(КЛИЕНТ.post("/predict", json={"review": ""}).status_code, 422)

    def test_текст_не_строка(self):
        self.assertEqual(КЛИЕНТ.post("/predict", json={"review": 12345}).status_code, 422)


class ТестСлужебных(unittest.TestCase):
    def test_model_info(self):
        ответ = КЛИЕНТ.get("/model-info")
        self.assertEqual(ответ.status_code, 200)
        self.assertIn("metrics", ответ.json())

    def test_docs_доступна(self):
        self.assertEqual(КЛИЕНТ.get("/docs").status_code, 200)


if __name__ == "__main__":
    unittest.main()