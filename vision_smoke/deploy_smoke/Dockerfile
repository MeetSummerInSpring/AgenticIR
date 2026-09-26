FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY smoke.py /app/smoke.py

CMD ["python", "/app/smoke.py"]
