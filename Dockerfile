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
RUN python -m pip install onnxruntime-gpu==1.26.0

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git ${FACEFUSION_PATH}

RUN python - <<'PY'
from pathlib import Path
p = Path("/opt/facefusion/facefusion/execution.py")
text = p.read_text()

old = """def create_value_and_unit(text : str) -> Optional[ValueAndUnit]:
\tif ' ' in text:
\t\tvalue, unit = text.split()
\t\treturn\
\t\t{
\t\t\t'value': int(value),
\t\t\t'unit': str(unit)
\t\t}
\treturn None
"""

new = """def create_value_and_unit(text : str) -> Optional[ValueAndUnit]:
\tif not text:
\t\treturn None
\ttext = str(text).strip()
\tif not text or text.lower().startswith('insufficient') or text.upper() == 'N/A':
\t\treturn None
\tparts = text.split()
\tif len(parts) >= 2:
\t\tvalue, unit = parts[0], parts[1]
\t\ttry:
\t\t\treturn\
\t\t\t{
\t\t\t\t'value': int(float(value)),
\t\t\t\t'unit': str(unit)
\t\t\t}
\t\texcept Exception:
\t\t\treturn None
\treturn None
"""

if old not in text:
    raise SystemExit("Expected create_value_and_unit block not found in execution.py")

p.write_text(text.replace(old, new))
print("Patched FaceFusion execution.py")
PY

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
