FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFUSION_PATH=/opt/facefusion \
    WORK_ROOT=/tmp/faceswap-jobs

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

ARG FACEFUSION_REF=master
RUN git clone --depth 1 --branch ${FACEFUSION_REF} https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

WORKDIR ${FACEFUSION_PATH}
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python install.py --onnxruntime cuda && \
    python -m pip install -r /tmp/requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
