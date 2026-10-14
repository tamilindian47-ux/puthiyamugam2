# IMPORTANT: Replace this with a real CUDA 12 Runpod base tag that exists.
# Examples of the *style* of tags you might find: cuda12.2.0, cuda12.4.0, cuda12.6.0, etc.
FROM runpod/base:0.4.0-cuda12.x.x

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps
# (Pin numpy to avoid ABI surprises; keep your original pins)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      "numpy==1.26.4" \
      cython && \
    python3 -m pip install \
      "onnxruntime-gpu==1.22.0" \
      "insightface==0.7.3" \
      opencv-python-headless \
      pillow \
      requests \
      tqdm \
      runpod

# Pre-download checkpoints into the image (reduces cold start)
RUN mkdir -p /app/checkpoints && \
    wget --progress=dot:giga --tries=5 --timeout=60 \
      -O /app/checkpoints/inswapper_128.onnx \
      "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

RUN wget --tries=5 --timeout=60 \
      -O /app/checkpoints/face_occluder.onnx \
      "https://huggingface.co/Rookiehan/facefusion/resolve/main/face_occluder.onnx" && \
    test -s /app/checkpoints/face_occluder.onnx

RUN wget --tries=5 --timeout=60 \
      -O /app/checkpoints/enhancer.onnx \
      "https://huggingface.co/untraining/facefusion/resolve/main/gpen_bfr_512.onnx" \
    || wget --tries=5 --timeout=60 \
      -O /app/checkpoints/enhancer.onnx \
      "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx" ; \
    test -s /app/checkpoints/enhancer.onnx

# InsightFace buffalo_l model
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -fL --retry 5 --retry-delay 5 \
      -o /tmp/buffalo_l.zip \
      "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
