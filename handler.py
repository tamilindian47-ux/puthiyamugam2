import os
import io
import cv2
import base64
import tempfile
import subprocess
import traceback
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageSequence
import runpod

INSWAPPER_PATH = os.environ.get("INSWAPPER_PATH", "/app/models/inswapper128.onnx")
GFPGAN_PATH = os.environ.get("GFPGAN_PATH", "/app/models/GFPGANv1.4.pth")

CLUSTER_THRESH = 0.45
MATCH_THRESH = 0.40
DETECT_SAMPLES = 30

PROVIDERS = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "HEURISTIC",
        },
    ),
    "CPUExecutionProvider",
]

print("Worker ORT providers:", ort.get_available_providers(), flush=True)
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider missing"

app = FaceAnalysis(
    name="buffalo_l",
    providers=PROVIDERS,
    allowed_modules=["detection", "recognition"],
)
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)

swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=PROVIDERS)
assert swapper.session.get_providers()[0] == "CUDAExecutionProvider", "Swapper fell back to CPU"

print("Worker Ready on GPU.", flush=True)

_gfpgan = None

def get_gfpgan():
    global _gfpgan
    if _gfpgan is None:
        try:
            from gfpgan import GFPGANer
            _gfpgan = GFPGANer(
                model_path=GFPGAN_PATH,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        except Exception as e:
            print("Worker GFPGAN unavailable:", e, flush=True)
            _gfpgan = False
    return _gfpgan

def b64_to_bytes(s):
    if "," in s:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)

def bytes_to_img(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def img_to_b64(img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")

def sniff_kind(data, isvideo_hint=False):
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[4:8] == b"ftyp" or data[:4] == b"\x1aE\xdf\xa3" or data[:4] == b"RIFF":
        return "video"
    return "video" if isvideo_hint else "image"

def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def gif_frames(data):
    img = Image.open(io.BytesIO(data))
    frames, durations = [], []
    for frame in ImageSequence.Iterator(img):
        rgb = frame.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        frames.append(bgr)
        durations.append(frame.info.get("duration", 80))
    return frames, durations

def write_temp(data, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path

def sample_video_frames(path, max_samples=30):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(max_samples, total)).astype(int).tolist()
    frames = []
    wanted = set(idxs)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(frame.copy())
        i += 1
    cap.release()
    return frames

def cluster_faces(frames):
    detections = []
    for frame in frames:
        faces = app.get(frame)
        h, w = frame.shape[:2]
        for f in faces:
            box = f.bbox.astype(int)
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(w, box[2]), min(h, box[3])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            detections.append({
                "embedding": np.asarray(f.embedding, dtype=np.float32),
                "normed_embedding": np.asarray(f.normed_embedding, dtype=np.float32),
                "thumb": crop,
                "score": float((x2 - x1) * (y2 - y1)),
            })

    if not detections:
        return []

    clusters = []
    for i, d in enumerate(detections):
        vec = d["normed_embedding"]
        matched = False
        for c in clusters:
            centroid = np.mean([detections[idx]["normed_embedding"] for idx in c], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            if float(np.dot(vec, centroid)) >= 1.0 - CLUSTER_THRESH:
                c.append(i)
                matched = True
                break
        if not matched:
            clusters.append([i])

    out = []
    for label, cidx in enumerate(clusters):
        best = max(cidx, key=lambda idx: detections[idx]["score"])
        centroid = np.mean([detections[idx]["embedding"] for idx in cidx], axis=0)
        thumbb64 = img_to_b64(detections[best]["thumb"], ".jpg")
        out.append({
            "character_id": int(label),
            "thumbnailbase64": thumbb64,
            "embedding": centroid.tolist(),
        })
    return out

def build_assignments(swaps):
    active = []
    for s in swaps:
        newface_b64 = s.get("newface")
        target_embedding = s.get("targetembedding")
        if not newface_b64 or target_embedding is None:
            continue
        srcimg = bytes_to_img(b64_to_bytes(newface_b64))
        if srcimg is None:
            continue
        srcfaces = app.get(srcimg)
        if not srcfaces:
            continue
        largest = max(
            srcfaces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        active.append({
            "targetemb": np.asarray(target_embedding, dtype=np.float32),
            "sourceface": largest,
        })
    return active

def maybe_enhance(frame, enable):
    if not enable:
        return frame
    enhancer = get_gfpgan()
    if not enhancer:
        return frame
    _, _, out = enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
    return out

def process_frame(img, assignments, enhance=False):
    faces = app.get(img)
    for tface in faces:
        best = None
        best_score = -1.0
        for s in assignments:
            sim = cosine(tface.embedding, s["targetemb"])
            if sim > best_score:
                best_score = sim
                best = s
        if best is not None and best_score >= MATCH_THRESH:
            img = swapper.get(img, tface, best["sourceface"], paste_back=True)
    return maybe_enhance(img, enhance)

def handle_detect(mediab64, isvideo):
    data = b64_to_bytes(mediab64)
    kind = sniff_kind(data, isvideo)

    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        frames = [img]
    elif kind == "gif":
        frames, _ = gif_frames(data)
    else:
        path = write_temp(data, ".mp4")
        try:
            frames = sample_video_frames(path, DETECT_SAMPLES)
        finally:
            if os.path.exists(path):
                os.remove(path)

    if not frames:
        return {"characters": []}

    return {"characters": cluster_faces(frames)}

def handle_swap(mediab64, isvideo, swaps, pipeline):
    assignments = build_assignments(swaps or [])
    if not assignments:
        return {"error": "No usable face found in the replacement photos"}

    enhance = str(pipeline).lower() not in ("", "none", "fast", "1")

    data = b64_to_bytes(mediab64)
    kind = sniff_kind(data, isvideo)

    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        out = process_frame(img, assignments, enhance)
        return {"mediatype": "image", "base64": img_to_b64(out)}

    if kind == "gif":
        frames, durations = gif_frames(data)
        out_frames = []
        for fr in frames:
            fr = process_frame(fr, assignments, enhance)
            out_frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        buf = io.BytesIO()
        out_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=out_frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )
        return {"mediatype": "gif", "base64": base64.b64encode(buf.getvalue()).decode("utf-8")}

    inpath = write_temp(data, ".mp4")
    rawpath = inpath + ".raw.mp4"
    finalpath = inpath + ".final.mp4"

    try:
        cap = cv2.VideoCapture(inpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(rawpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(process_frame(fr, assignments, enhance))

        cap.release()
        writer.release()

        cmd = [
            "ffmpeg", "-y",
            "-i", rawpath,
            "-i", inpath,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "faststart",
            finalpath
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        target = finalpath if r.returncode == 0 and os.path.exists(finalpath) else rawpath

        with open(target, "rb") as f:
            outb64 = base64.b64encode(f.read()).decode("utf-8")
        return {"mediatype": "video", "base64": outb64}
    finally:
        for p in (inpath, rawpath, finalpath):
            if p and os.path.exists(p):
                os.remove(p)

def handler(job):
    job_input = job.get("input", {}) or {}
    action = job_input.get("action")
    mediab64 = job_input.get("media") or job_input.get("targetimage") or job_input.get("target")
    isvideo = bool(job_input.get("isvideo", False))

    if not mediab64:
        return {
            "error": "Missing media payload for detection.",
            "keysreceived": list(job_input.keys())
        }

    try:
        if action == "detect":
            return handle_detect(mediab64, isvideo)
        if action == "swap":
            return handle_swap(
                mediab64,
                isvideo,
                job_input.get("swaps", []),
                job_input.get("pipeline", "fast"),
            )
        return {"error": f"Invalid action: {action}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
