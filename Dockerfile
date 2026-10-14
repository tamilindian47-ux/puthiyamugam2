FROM runpod/base:0.4.0-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FACEFUSION_PATH=/opt/facefusion

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 wget curl unzip git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
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
        opencv-python-headless pillow requests tqdm runpod scipy filetype

# Clone FaceFusion core for the CLI processors
RUN git clone --depth 1 https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
