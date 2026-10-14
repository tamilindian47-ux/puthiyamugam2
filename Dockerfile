FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFUSION_PATH=/opt/facefusion \
    WORK_ROOT=/tmp/faceswap-jobs \
    FF_EXECUTION_PROVIDER=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
