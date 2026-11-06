Учебный проект практики 4: модель как API-сервис, упакованный в контейнер.

Что внутри

Файл train.py: обучение модели (TF-IDF + LogReg) и упаковка артефактов
Файл app/main.py: сервис на FastAPI: /health, /predict, /model-info
Файл app/schemas.py: схема запроса и ответа, валидация входа
Файл app/preprocessing.py: очистка HTML-разметки отзыва
Файл tests/test_api.py: тесты сервиса
Файл tests/test_model.py: тесты модели и артефактов
Папка data/: данные, файл IMDB Dataset.csv (50 000 строк)
Папка artifacts/: модель и метаданные, создаются обучением
Файл Dockerfile: образ сервиса
Файл requirements.txt: зависимости


Как запустить локально

1. Установить зависимости:
   python -m pip install -r requirements.txt
2. Обучить модель:
   python train.py
3. Запустить сервис:
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8077
4. Проверить:
   curl http://127.0.0.1:8077/health



Как проверить тестами

python -m unittest discover -s tests -v


Как собрать образ

docker build -t kinootzyv-service:1.0.0 .
docker run --rm -p 8077:8077 kinootzyv-service:1.0.0
