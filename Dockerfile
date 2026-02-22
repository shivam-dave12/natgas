# ================================================
# Dockerfile — Natural Gas Predictor v18 Daemon
# ================================================

FROM python:3.12-slim

# Install system build dependencies (needed for xgboost, lightgbm, arch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy both Python scripts
COPY natgas_daemon.py .
COPY natural_gas_predictor_enhanced*.py ./
COPY .env .

# Create directories for logs & persistence
RUN mkdir -p /app/logs /app/data

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Volume for .env, logs, and any persistent data
VOLUME ["/app"]

# Default command (you can override with flags)
# Examples:
#   --run-once
#   --no-telegram
#   --auto-start
CMD ["python", "natgas_daemon.py"]
