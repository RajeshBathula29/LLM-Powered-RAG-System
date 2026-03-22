# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: final image ──────────────────────────────────────────────────────
FROM deps AS final

# Copy source
COPY . .

# Persist FAISS index across restarts
VOLUME ["/app/vectorstore"]

# Data directory for uploaded documents
VOLUME ["/app/data"]

EXPOSE 5000

ENV FLASK_ENV=production \
    PORT=5000

# Gunicorn for production (multi-worker)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "app.api:app"]
