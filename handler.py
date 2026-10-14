import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np
import requests
import runpod
from insightface.app import FaceAnalysis

FF_DIR = os.getenv("FACEFUSION_PATH", "/opt/facefusion")
WORK_ROOT = os.getenv("WORK_ROOT", "/tmp/faceswap-jobs")
FF_TIMEOUT = int(os.getenv("FF_TIMEOUT", "1800"))
EXECUTION_PROVIDER = os.getenv("FF_EXECUTION_PROVIDER", "cuda")
DETECT_SIZE = tuple(int(x) for x in os.getenv("DETECT_SIZE", "640,640").split(","))
MAX_DETECT_FACES = int(os.getenv("MAX_DETECT_FACES", "12"))
SAMPLE_FRAMES = int(os.getenv("SAMPLE_FRAMES", "12"))
CLUSTER_THRESHOLD = float(os.getenv("CLUSTER_THRESHOLD", "0.58"))
REFERENCE_FACE_DISTANCE = float(os.getenv("REFERENCE_FACE_DISTANCE", "0.6"))
os.makedirs(WORK_ROOT, exist_ok=True)

MODELS = {
    "inswapper": "inswapper_128", "inswapper_128": "inswapper_128",
    "hyperswap_1a": "hyperswap_1a_256", "hyperswap_1a_256": "hyperswap_1a_256",
    "hyperswap_1b": "hyperswap_1b_256", "hyperswap_1b_256": "hyperswap_1b_256",
    "hyperswap_1c": "hyperswap_1c_256", "hyperswap_1c_256": "hyperswap_1c_256",
}

_face_app = None


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
            text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as e:
        return f"unavailable: {e}"


def sources_from(inp):
    if inp.get("sources"):
        return inp["sources"]
    if inp.get("source"):
        return [inp["source"]]
    swaps = inp.get("swaps") or []
    return [s["new_face"] for s in swaps if s.get("new_face")]


def get_face_app():
    global _face_app
    if _face_app is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if EXECUTION_PROVIDER == "cuda" else ["CPUExecutionProvider"]
        _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        _face_app.prepare(ctx_id=0 if EXECUTION_PROVIDER == "cuda" else -1, det_size=DETECT_SIZE)
    return _face_app


def norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine(a, b):
    return float(np.dot(norm(a), norm(b)))


def jpeg_b64(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode thumbnail")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def face_center(face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def union_find_cluster(items, threshold):
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if cosine(items[i]["embedding"], items[j]["embedding"]) >= threshold:
                union(i, j)

    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[find(i)].append(item)
    return list(groups.values())


def sample_video_frames(path, limit):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for detection")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_indexes = [0] if total <= 1 else sorted(set(np.linspace(0, max(total - 1, 0), num=min(limit, total), dtype=int).tolist()))
    frames = []
    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((idx, frame))
    cap.release()
    return frames


def detect_faces_image(frame):
    app = get_face_app()
    faces = app.get(frame) or []
    faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2.0)
    return faces[:MAX_DETECT_FACES]


def build_characters_from_image(frame):
    faces = detect_faces_image(frame)
    chars = []
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame.shape[:2]
        pad = int(max(x2 - x1, y2 - y1) * 0.18)
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[ya:yb, xa:xb]
        chars.append({
            "position": idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "embedding": norm(face.normed_embedding).tolist(),
            "thumbnail_base64": jpeg_b64(crop)
        })
    return chars


def build_characters_from_video(path):
    samples = []
    for frame_idx, frame in sample_video_frames(path, SAMPLE_FRAMES):
        for face in detect_faces_image(frame):
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = frame.shape[:2]
            pad = int(max(x2 - x1, y2 - y1) * 0.18)
            xa, ya = max(0, x1 - pad), max(0, y1 - pad)
            xb, yb = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[ya:yb, xa:xb]
            samples.append({
                "frame": frame_idx,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": face_center(face),
                "embedding": norm(face.normed_embedding),
                "thumbnail_base64": jpeg_b64(crop),
                "area": float((x2 - x1) * (y2 - y1))
            })
    if not samples:
        return []
    groups = union_find_cluster(samples, CLUSTER_THRESHOLD)
    groups = sorted(groups, key=lambda g: max(item["area"] for item in g), reverse=True)
    characters = []
    for group in groups[:MAX_DETECT_FACES]:
        best = max(group, key=lambda item: item["area"])
        avg_x = sum(item["center"][0] for item in group) / len(group)
        mean_embedding = norm(np.mean([item["embedding"] for item in group], axis=0))
        characters.append({
            "sort_x": avg_x,
            "position": 0,
            "bbox": best["bbox"],
            "embedding": mean_embedding.tolist(),
            "thumbnail_base64": best["thumbnail_base64"]
        })
    characters = sorted(characters, key=lambda c: c["sort_x"])
    for idx, c in enumerate(characters):
        c["position"] = idx
        c.pop("sort_x", None)
    return characters


def facefusion_command(source_path, target, output, temp_dir, inp, selector_mode, reference_face_position=None):
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
        "--source-paths", source_path,
        "--target-path", target,
        "--output-path", output,
        "--temp-path", temp_dir,
        "--processors", *processors,
        "--face-swapper-model", model,
        "--face-selector-mode", selector_mode,
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
    if selector_mode == "reference":
        cmd += [
            "--reference-face-position", str(int(reference_face_position or 0)),
            "--reference-frame-number", str(int(inp.get("reference_frame_number", 0))),
            "--reference-face-distance", str(float(inp.get("reference_face_distance", REFERENCE_FACE_DISTANCE)))
        ]
    return cmd, model


def run_facefusion(command):
    t0 = time.time()
    result = subprocess.run(command, cwd=FF_DIR, text=True, capture_output=True, timeout=FF_TIMEOUT)
    seconds = round(time.time() - t0, 2)
    logs = (result.stdout + "\n" + result.stderr)[-6000:]
    return result.returncode, seconds, logs


def do_detect(inp):
    media = inp.get("media")
    if not media:
        return {"status": "failed", "error": "Provide media."}
    is_video = bool(inp.get("is_video", True))
    folder = tempfile.mkdtemp(prefix="faceswap_detect_", dir=WORK_ROOT)
    try:
        target = save_input(media, folder, "target", ".mp4" if is_video else ".jpg")
        if is_video:
            fixed = os.path.join(folder, "target_fixed.mp4")
            normalize_video(target, fixed)
            chars = build_characters_from_video(fixed)
        else:
            frame = cv2.imread(target)
            if frame is None:
                raise RuntimeError("Unable to open image for detection")
            chars = build_characters_from_image(frame)
        return {"status": "success", "characters": chars}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def do_swap(inp):
    media = inp.get("media")
    swaps = inp.get("swaps") or []
    source = inp.get("source")
    sources = inp.get("sources") or []
    if not media:
        return {"status": "failed", "error": "Provide media."}
    if not swaps and not source and not sources:
        return {"status": "failed", "error": "Provide swaps, source, or sources."}
    is_video = bool(inp.get("is_video", True))
    folder = tempfile.mkdtemp(prefix="faceswap_swap_", dir=WORK_ROOT)
    try:
        target = save_input(media, folder, "target", ".mp4" if is_video else ".jpg")
        if is_video:
            fixed = os.path.join(folder, "target_fixed.mp4")
            normalize_video(target, fixed)
            target = fixed
        current_target = target
        total_seconds = 0.0
        model_used = None
        last_logs = ""

        if swaps:
            ordered_swaps = sorted(swaps, key=lambda s: int(s.get("position", 0)))
            for idx, swap in enumerate(ordered_swaps):
                source_path = save_input(swap["new_face"], folder, f"source_{idx}", ".jpg")
                out_path = os.path.join(folder, f"pass_{idx}.{'mp4' if is_video else 'jpg'}")
                cmd, model_used = facefusion_command(
                    source_path,
                    current_target,
                    out_path,
                    os.path.join(folder, f"temp_{idx}"),
                    inp,
                    selector_mode="reference",
                    reference_face_position=int(swap.get("position", idx))
                )
                code, seconds, logs = run_facefusion(cmd)
                total_seconds += seconds
                last_logs = logs
                if code != 0 or not os.path.isfile(out_path):
                    return {"status": "failed", "error": "FaceFusion failed", "returncode": code, "log_tail": logs}
                current_target = out_path
        else:
            src_list = sources or ([source] if source else [])
            source_path = save_input(src_list[0], folder, "source_0", ".jpg")
            out_path = os.path.join(folder, f"output.{'mp4' if is_video else 'jpg'}")
            cmd, model_used = facefusion_command(
                source_path,
                current_target,
                out_path,
                os.path.join(folder, "temp_0"),
                inp,
                selector_mode=str(inp.get("selector_mode", "many")),
                reference_face_position=int(inp.get("reference_face_position", 0))
            )
            code, seconds, logs = run_facefusion(cmd)
            total_seconds += seconds
            last_logs = logs
            if code != 0 or not os.path.isfile(out_path):
                return {"status": "failed", "error": "FaceFusion failed", "returncode": code, "log_tail": logs}
            current_target = out_path

        with open(current_target, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
        return {
            "status": "success",
            "media_type": "video" if is_video else "image",
            "base64": payload,
            "metrics": {
                "model": model_used,
                "seconds": round(total_seconds, 2),
                "output_mb": round(os.path.getsize(current_target) / 1048576, 2)
            },
            "log_tail": last_logs[-1000:]
        }
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def handler(event):
    inp = event.get("input", {}) or {}
    try:
        action = inp.get("action")
        if action == "info":
            return {"status": "success", "gpu": gpu_state(), "onnxruntime": onnx_state(), "models": list(MODELS)}
        if action == "detect":
            return do_detect(inp)
        return do_swap(inp)
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


runpod.serverless.start({"handler": handler})
