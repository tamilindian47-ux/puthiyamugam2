# syntax=docker/dockerfile:1

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
import re

p = Path("/opt/facefusion/facefusion/execution.py")
text = p.read_text()

pattern = r"def create_value_and_unit\\(text\\s*:\\s*str\\)\\s*->\\s*Optional\\[ValueAndUnit\\]:\\n(?:\\t|    ).*?(?=\\ndef |\\nclass |\\Z)"

replacement = '''def create_value_and_unit(text : str) -> Optional[ValueAndUnit]:
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
'''

new_text, count = re.subn(pattern, replacement, text, flags=re.S)
if count != 1:
    raise SystemExit(f"Expected to patch 1 function, patched {count}")

p.write_text(new_text)
print("Patched FaceFusion execution.py")
PY

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
