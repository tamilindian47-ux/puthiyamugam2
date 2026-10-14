FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ffmpeg libgl1 libglib2.0-0 build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/facefusion/facefusion.git /app/facefusion
WORKDIR /app/facefusion

RUN pip install --upgrade pip && \
    pip install -r requirements.txt || true && \
    pip install "numpy<2.0.0" --force-reinstall

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
