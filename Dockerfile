FROM python:3.10-slim

# System libs OpenCV/tensorflow may need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the runtime deps for the API
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# copy from notebooks/api to /app/api
COPY notebooks/api/ ./api/


ENV MODEL_PATH=api/model/brain_mri_model.h5
ENV THRESHOLD=0.05
ENV PORT=8000

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "api.app:app"]
