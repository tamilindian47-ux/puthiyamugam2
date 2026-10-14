FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    FACEFUSION_PATH=/opt/facefusion \
    WORK_ROOT=/tmp/faceswap-jobs

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg curl ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Clone FaceFusion source
ARG FACEFUSION_REF=master
RUN git clone --depth 1 --branch ${FACEFUSION_REF} https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

WORKDIR ${FACEFUSION_PATH}

# 1. Upgrade packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch with CUDA 12.4 support
RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install onnxruntime-gpu and core ML packages
RUN python -m pip install \
    "onnxruntime-gpu>=1.19.0" \
    opencv-python-headless \
    numpy \
    pillow \
    tqdm \
    scipy \
    filetype

# 4. Install FaceFusion package requirements directly
RUN if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi

# 5. Install RunPod serverless worker requirements
COPY requirements.txt /tmp/runpod_requirements.txt
RUN python -m pip install -r /tmp/runpod_requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
