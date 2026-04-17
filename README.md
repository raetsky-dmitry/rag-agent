# RAG-агент для интернет-магазина "ЭлектроМощь"

Этот проект представляет собой RAG-агента (Retrieval-Augmented Generation) для консультации клиентов интернет-магазина электроники "ЭлектроМощь".

## Требования

- Python 3.10 или выше
- API-ключ OpenRouter

## Установка

1. Клонируйте репозиторий:

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd rag-agent
```

2. Создайте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # На Linux/Mac
# или
venv\Scripts\activate  # На Windows
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` в корне проекта и добавьте свои API-ключи:

```env
OPENROUTER_API_KEY=ваш_api_ключ_от_openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME_QWEN=qwen/qwen3.5-122b-a10b
MODEL_NAME_DEEPSEEK=deepseek/deepseek-v3.2
MODEL_NAME_FREE=openrouter/free
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ваш_api_ключ_langsmith
LANGSMITH_PROJECT=rag-agent
```

## Запуск

Для запуска агента в режиме чата выполните команду:

```bash
python main.py
```

Для запуска агента в режиме API выполните команду:

```bash
python -m uvicorn api:app --reload
```

Запустится сервер. После этого:

- Открйте в браузере указанный при активации адрес (например, http://localhost:8000)
- Перейдите в папку docs (http://localhost:8000/docs)
- Откройте [POST] /chat
- Нажмите кнопку "Try it out"

## Архитектура

Проект состоит из следующих модулей:

- `agent/` - содержит логику агента, инструменты и промпты
- `rag/` - содержит компоненты RAG (чтение документов, разбиение на части, эмбеддинги)
- `chat/` - содержит логику чата и сессий
- `app/` - содержит настройки приложения
- `data/_docs/` - содержит документы базы знаний

## Возможности

- Поиск товаров по названию, цене и наличию
- Проверка сроков доставки в указанный город
- Добавление товаров в корзину
- Получение информации о товарах
- Расчет итоговой суммы корзины
- Поиск по базе знаний магазина
