# SynthGuard-Ops pipeline image (CPU only). Multi-arch base: works on amd64 and Apple Silicon.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BIN=/usr/local/bin \
    PY=/usr/local/bin/python \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

WORKDIR /app

# 1) CPU-only torch from the PyTorch CPU index (avoids multi-GB CUDA wheels on amd64).
# 2) Everything else pinned by requirements.lock (torch line removed: already installed).
COPY requirements.lock .
RUN TORCH=$(grep -E '^torch==' requirements.lock) \
    && pip install "$TORCH" --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^torch==' requirements.lock > /tmp/req.txt \
    && pip install -r /tmp/req.txt

# .git is not copied into the image; record the commit for MLflow tags instead.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT

# Project code, installed editable so a bind-mounted /app is used as-is.
COPY . .
RUN pip install --no-deps -e .

RUN useradd --create-home --uid 1000 synthguard && chown -R synthguard /app
USER synthguard

CMD ["make", "test"]
