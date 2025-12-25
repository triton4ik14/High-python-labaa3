FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir numpy matplotlib

# Копируем код
COPY . .

# Команда запуска
CMD ["python", "main.py"]
