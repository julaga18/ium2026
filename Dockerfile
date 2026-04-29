FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    MLFLOW_TRACKING_URI=file:/app/mlruns

RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    pandas \
    scikit-learn \
    seaborn \
    joblib \
    kagglehub \
    mlflow==2.12.1 \
    numpy

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app/mlruns /tmp/torchinductor

RUN useradd -u 1003 -m jenkins

COPY . /app

RUN chmod +x scripts/download_and_process.sh

RUN chown -R 1003:1003 /app /tmp/torchinductor /app/mlruns

USER jenkins

CMD ["/bin/bash"]