# ==============================================================================
# Backend Dockerfile - Adaptive Threat Intelligence NIDS
# Python 3.11 with OpenMP support for XGBoost / Scikit-Learn
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenMP and scientific compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application packages and serialized models
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV HIDE_INTERNAL_ERRORS=true

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
