# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential git  && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -U pip  && pip install --no-cache-dir -e ".[ui]"

ENV STREAMLIT_SERVER_PORT=8501     STREAMLIT_SERVER_ADDRESS=0.0.0.0     PYTHONUNBUFFERED=1

RUN mkdir -p /app/outputs /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
