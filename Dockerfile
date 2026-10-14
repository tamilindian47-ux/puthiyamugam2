FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFUSION_PATH=/opt/facefusion \
    WORK_ROOT=/tmp/faceswap-jobs \
    FF_EXECUTION_PROVIDER=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ffmpeg git curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -f /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python --version && python -m pip --version && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
