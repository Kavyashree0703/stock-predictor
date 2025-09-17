# Dockerfile for Flask backend
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy and install dependencies
COPY requirements-backend.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# expose port
EXPOSE 5000

# run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120"]
