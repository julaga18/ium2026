FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pandas kaggle

# --- KONFIGURACJA KAGGLE ---
# 1. Tworzymy wymagany folder w katalogu domowym użytkownika root
RUN mkdir -p /root/.config/kaggle

# 2. Kopiujemy plik z Twojego komputera do kontenera
COPY kaggle.json /root/.config/kaggle/kaggle.json

# 3. Nadajemy uprawnienia (tylko odczyt/zapis dla właściciela)
RUN chmod 600 /root/.config/kaggle/kaggle.json
# ---------------------------

WORKDIR /app
COPY . .

CMD ["/bin/bash"]