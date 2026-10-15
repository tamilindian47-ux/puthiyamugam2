FROM runpod/base:0.4.0-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "numpy==1.26.4" cython && \
    python3 -m pip install --no-cache-dir \
        "onnxruntime-gpu==1.22.0" \
        "nvidia-cuda-runtime-cu12" \
        "nvidia-cuda-nvrtc-cu12" \
        "nvidia-cublas-cu12" \
        "nvidia-cufft-cu12" \
        "nvidia-curand-cu12" \
        "nvidia-cudnn-cu12==9.*" \
        insightface==0.7.3 \
        opencv-python-headless pillow requests tqdm runpod && \
    python3 -m pip install --no-cache-dir "numpy==1.26.4"

RUN mkdir -p /app/checkpoints && \
    wget --progress=dot:giga --tries=5 --timeout=60 \
      -O /app/checkpoints/inswapper_128.onnx \
      "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

RUN wget --tries=5 --timeout=60 -O /app/checkpoints/face_occluder.onnx \
      "https://huggingface.co/Rookiehan/facefusion/resolve/main/face_occluder.onnx" \
    || wget --tries=5 --timeout=60 -O /app/checkpoints/face_occluder.onnx \
      "https://huggingface.co/JackCui/facefusion/resolve/main/face_occluder.onnx" \
    || wget --tries=5 --timeout=60 -O /app/checkpoints/face_occluder.onnx \
      "https://github.com/facefusion/facefusion-assets/releases/download/models/face_occluder.onnx"; \
    test -s /app/checkpoints/face_occluder.onnx

RUN wget --tries=5 --timeout=60 -O /app/checkpoints/enhancer.onnx \
      "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx" \
    || wget --tries=5 --timeout=60 -O /app/checkpoints/enhancer.onnx \
      "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx"; \
    test -s /app/checkpoints/enhancer.onnx

RUN mkdir -p /root/.insightface/models/buffalo_l && \
    ( curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" || \
      curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://huggingface.co/techwavelaps/aceplus-face/resolve/main/insightface/models/buffalo_l.zip" ) && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
