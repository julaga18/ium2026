FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch pandas scikit-learn seaborn joblib kagglehub

RUN mkdir -p /tmp/torchinductor

RUN useradd -u 1003 -m jenkins

COPY . /app

RUN chmod +x scripts/download_and_process.sh

CMD ["/bin/bash"]