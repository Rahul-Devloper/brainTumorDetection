FROM python:3.10-slim

# System deps (wget for HEALTHCHECK; libgomp for TF; pillow uses zlib/libjpeg via wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy app code (make sure your model file is included below)
COPY . .

# Lightweight readiness probe (Render doesn’t require it, but it helps)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://127.0.0.1:${PORT}/health || exit 1

# IMPORTANT: Bind to $PORT (do NOT hard-code 8000)
CMD ["sh", "-c", "gunicorn notebooks.api.app:app -b 0.0.0.0:$PORT --workers 1 --threads 2"]