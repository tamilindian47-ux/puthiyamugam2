FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFUSION_PATH=/opt/facefusion \
    WORK_ROOT=/tmp/faceswap-jobs \
    FF_EXECUTION_PROVIDER=cuda \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python --version && python -c "from typing import NotRequired; print('NotRequired OK')"

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install numpy requests runpod opencv-python-headless
RUN python -m pip install insightface==0.7.3
RUN python -m pip uninstall -y onnxruntime onnxruntime-gpu || true
RUN python -m pip install onnxruntime-gpu

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
