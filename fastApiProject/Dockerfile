FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements и ставим зависимости
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY app/ .

# Создаём необходимые папки
RUN mkdir -p tmp models fonts certs static templates

# Пользователь без root (безопасность)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# По умолчанию запускаем без SSL (удобно для Docker)
# Если нужны сертификаты — раскомментируйте нижний вариант
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Вариант с SSL (если сертификаты лежат в /app/certs)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
#      "--ssl-keyfile", "certs/key.pem", "--ssl-certfile", "certs/cert.pem"]