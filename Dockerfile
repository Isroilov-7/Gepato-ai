FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Audio: libsndfile (soundfile), ffmpeg (brauzer webm/m4a yozuvlari). Praat — parselmouth ichida
# (alohida `praat` GUI paketi ortiqcha edi — kod uni chaqirmaydi). gcc — wheel bo'lmasa build uchun.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads models

# --reload YO'Q (productionda fayl kuzatuvchisi CPU/xotira yeydi va kodni tasodifan qayta yuklaydi)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
