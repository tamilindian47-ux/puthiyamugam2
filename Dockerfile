FROM runpod/base:0.4.0-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
    "numpy==1.26.4" \
    "onnxruntime-gpu==1.17.1" \
    "insightface==0.7.3" \
    "opencv-python-headless==4.9.0.80" \
    "pillow>=10.0.0" \
    "imageio>=2.31.0" \
    "imageio-ffmpeg>=0.4.9" \
    requests \
    tqdm \
    runpod

RUN mkdir -p /app/checkpoints && \
    wget --progress=dot:giga --tries=5 --timeout=60 \
      -O /app/checkpoints/inswapper_128.onnx \
      https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

RUN mkdir -p /root/.insightface/models && \
    curl -fL --retry 5 --retry-delay 5 \
      -o /tmp/buffalo_l.zip \
      https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
