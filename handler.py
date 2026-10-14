import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import requests
import runpod

FF_DIR = os.getenv("FACEFUSION_PATH", "/opt/facefusion")
WORK_ROOT = os.getenv("WORK_ROOT", "/tmp/faceswap-jobs")
FF_TIMEOUT = int(os.getenv("FF_TIMEOUT", "1800"))
EXECUTION_PROVIDER = os.getenv("FF_EXECUTION_PROVIDER", "cuda")
os.makedirs(WORK_ROOT, exist_ok=True)

MODELS = {
    "inswapper": "inswapper_128", "inswapper_128": "inswapper_128",
    "hyperswap_1a": "hyperswap_1a_256", "hyperswap_1a_256": "hyperswap_1a_256",
    "hyperswap_1b": "hyperswap_1b_256", "hyperswap_1b_256": "hyperswap_1b_256",
    "hyperswap_1c": "hyperswap_1c_256", "hyperswap_1c_256": "hyperswap_1c_256",
}

def sniff_ext(blob, default_ext):
    if blob[:3] == b"\xff\xd8\xff": return ".jpg"
    if blob[:8] == b"\x89PNG\r\n\x1a\n": return ".png"
    if blob[:4] == b"GIF8": return ".gif"
    if len(blob) >= 8 and blob[4:8] == b"ftyp": return ".mp4"
    if blob[:4] == b"\x1a\x45\xdf\xa3": return ".webm"
    return default_ext

def get_bytes(value):
    if not isinstance(value, str):
        raise ValueError("Media must be a base64/data URL string or an HTTPS URL.")
    if value.startswith(("https://", "http://")):
        r = requests.get(value, timeout=120)
        r.raise_for_status()
        return r.content
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)

def save_input(value, folder, name, default_ext):
    blob = get_bytes(value)
    path = os.path.join(folder, name + sniff_ext(blob, default_ext))
    with open(path, "wb") as f:
        f.write(blob)
    return path

def gpu_state():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=15
        ).strip()
    except Exception as e:
        return f"unavailable: {e}"

def onnx_state():
    try:
        return subp
