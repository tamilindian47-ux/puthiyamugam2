FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ffmpeg libgl1 libglib2.0-0 build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone FaceFusion if needed for other scripts
RUN git clone https://github.com/facefusion/facefusion.git /app/facefusion

# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Create model destination directories
RUN mkdir -p /app/checkpoints /root/.insightface/models/buffalo_l

# 1. Download Inswapper 128
RUN wget --progress=dot:giga --tries=5 --timeout=60 \
    https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx \
    -O /app/checkpoints/inswapper_128.onnx

# 2. Download Face Occluder (using verified models-3.0.0 tag)
RUN wget --progress=dot:giga --tries=5 --timeout=60 \
    https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/face_occluder.onnx \
    -O /app/checkpoints/face_occluder.onnx

# 3. Download CodeFormer Enhancer (using verified models-3.0.0 tag)
RUN wget --progress=dot:giga --tries=5 --timeout=60 \
    https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx \
    -O /app/checkpoints/enhancer.onnx

# 4. Bake Buffalo_L so insightface never calls out to GitHub at startup
RUN wget --progress=dot:giga --tries=5 --timeout=60 \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    -O /tmp/buffalo_l.zip && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
