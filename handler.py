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


def normalize_video(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst],
        check=True, timeout=600)


def gpu_state():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=15).strip()
    except Exception as e:
        return f"unavailable: {e}"


def onnx_state():
    try:
        return subprocess.check_output(
            [sys.executable, "-c", "import onnxruntime as ort; print(ort.get_available_providers())"],
            cwd=FF_DIR, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as e:
        return f"unavailable: {e}"


def sources_from(inp):
    if inp.get("sources"):
        return inp["sources"]
    if inp.get("source"):
        return [inp["source"]]
    swaps = inp.get("swaps") or []
    if swaps and swaps[0].get("new_face"):
        return [swaps[0]["new_face"]]
    return []


def facefusion_command(sources, target, output, temp_dir, inp):
    requested = str(inp.get("model", "hyperswap_1c")).lower()
    model = MODELS.get(requested)
    if not model:
        raise ValueError("Use inswapper_128, hyperswap_1a, hyperswap_1b, or hyperswap_1c.")
    expression = bool(inp.get("expression", True))
    enhancer = str(inp.get("enhancer", "none")).lower()
    processors = ["face_swapper"]
    if expression:
        processors.append("expression_restorer")
    if enhancer != "none":
        processors.append("face_enhancer")
    cmd = [
        sys.executable, "facefusion.py", "headless-run",
        "--source-paths", *sources,
        "--target-path", target,
        "--output-path", output,
        "--temp-path", temp_dir,
        "--processors", *processors,
        "--face-swapper-model", model,
        "--face-selector-mode", str(inp.get("selector_mode", "many")),
        "--face-mask-types", "box", "occlusion",
        "--face-mask-blur", str(float(inp.get("mask_blur", 0.3))),
        "--execution-providers", EXECUTION_PROVIDER,
        "--output-video-preset", str(inp.get("video_preset", "fast")),
        "--output-video-quality", str(int(inp.get("video_quality", 90))),
        "--output-image-quality", str(int(inp.get("image_quality", 95))),
        "--log-level", "info",
    ]
    if expression:
        cmd += ["--expression-restorer-model", "live_portrait",
                "--expression-restorer-factor", str(int(inp.get("expression_factor", 80)))]
        areas = inp.get("expression_areas") or []
        if areas:
            cmd += ["--expression-restorer-areas", *areas]
    if enhancer != "none":
        cmd += ["--face-enhancer-model", enhancer,
                "--face-enhancer-blend", str(int(inp.get("enhancer_blend", 40)))]
    if inp.get("pixel_boost"):
        cmd += ["--face-swapper-pixel-boost", str(inp["pixel_boost"])]
    if inp.get("selector_mode") == "reference":
        cmd += ["--reference-face-position", str(int(inp.get("reference_face_position", 0))),
                "--reference-frame-number", str(int(inp.get("reference_frame_number", 0)))]
    return cmd, model


def do_swap(inp):
    sources = sources_from(inp)
    media = inp.get("media")
    if not media or not sources:
        return {"status": "failed", "error": "Provide media and source, sources, or swaps[0].new_face."}
    is_video = bool(inp.get("is_video", True))
    folder = tempfile.mkdtemp(prefix="faceswap_", dir=WORK_ROOT)
    try:
        source_paths = [save_input(v, folder, f"source_{i}", ".jpg") for i, v in enumerate(sources)]
        target = save_input(media, folder, "target", ".mp4" if is_video else ".jpg")
        if is_video:
            fixed = os.path.join(folder, "target_fixed.mp4")
            normalize_video(target, fixed)
            target = fixed
        output = os.path.join(folder, "output.mp4" if is_video else "output.jpg")
        command, model = facefusion_command(source_paths, target, output, os.path.join(folder, "temp"), inp)
        t0 = time.time()
        result = subprocess.run(command, cwd=FF_DIR, text=True, capture_output=True, timeout=FF_TIMEOUT)
        seconds = round(time.time() - t0, 2)
        logs = (result.stdout + "\n" + result.stderr)[-4000:]
        if result.returncode != 0 or not os.path.isfile(output):
            return {"status": "failed", "error": "FaceFusion failed", "returncode": result.returncode, "log_tail": logs}
        with open(output, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
        return {"status": "success", "media_type": "video" if is_video else "image", "base64": payload,
                "metrics": {"model": model, "seconds": seconds,
                            "output_mb": round(os.path.getsize(output) / 1048576, 2)}}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def handler(event):
    inp = event.get("input", {}) or {}
    try:
        if inp.get("action") == "info":
            return {"status": "success", "gpu": gpu_state(), "onnxruntime": onnx_state(), "models": list(MODELS)}
        if inp.get("action") == "detect":
            return {"status": "failed", "error": "detect is not supported on this test endpoint; use the Face Swap Lab page."}
        return do_swap(inp)
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


runpod.serverless.start({"handler": handler})
