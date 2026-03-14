FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY voice_worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY voice_worker ./voice_worker

EXPOSE 8010

CMD ["uvicorn", "voice_worker.app.main:app", "--host", "0.0.0.0", "--port", "8010"]
