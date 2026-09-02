FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY deployment/model ./deployment/model

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "arabic_sentiment.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]