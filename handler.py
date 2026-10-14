import os
import io
import cv2
import base64
import tempfile
import subprocess
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageSequence
import runpod

INSWAPPERPATH = os.environ.get("INSWAPPERPATH", "models/inswapper128.onnx")
GFPGANPATH = "models/GFPGANv1.4.pth"

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

swapper = insightface.model_zoo.get_model(INSWAPPERPATH, providers=PROVIDERS)
assert swapper.session.get_providers()[0] == "CUDAExecutionProvider", "Swapper fell back to CPU"

print("Worker Ready on GPU.", flush=True)

gfpgan = None


def get_gfpgan():
    global gfpgan
    if gfpgan is None:
        try:
            from gfpgan import GFPGANer

            gfpgan = GFPGANer(
                model_path=GFPGANPATH,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        except Exception as e:
            print("Worker GFPGAN unavailable:", e, flush=True)
            gfpgan = False
    return gfpgan


def b64_to_bytes(s):
    if "," in s[:100]:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def bytes_to_img(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def img_to_b64(img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def sniff_kind(data, isvideo_hint):
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[4:8] == b"ftyp" or data[:4] == b"\u001AEß£" or data[:4] == b"RIFF":
        return "video"
    return "video" if isvideo_hint else "image"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def write_temp(data, suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return f.name


def gif_frames(data):
    im = Image.open(io.BytesIO(data))
    frames = []
    durations = []
    for fr in ImageSequence.Iterator(im):
        durations.append(fr.info.get("duration", 80))
        frames.append(cv2.cvtColor(np.array(fr.convert("RGB")), cv2.COLOR_RGB2BGR))
    return frames, durations


def sample_video_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []

    if total > 0:
        idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
    else:
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)

    cap.release()
    return frames


def crop_face(img, face):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return img[y1:y2, x1:x2]


def cluster_faces(frames):
    clusters = []

    for frame in frames:
        faces = app.get(frame)
        for f in faces:
            emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)
            matched = False

            for cluster in clusters:
                centroid = np.mean([x["embedding"] for x in cluster], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                if float(np.dot(emb, centroid)) >= (1.0 - CLUSTER_THRESH):
                    cluster.append(
                        {
                            "embedding": emb,
                            "face": f,
                            "crop": crop_face(frame, f),
                            "score": float(
                                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                            ),
                        }
                    )
                    matched = True
                    break

            if not matched:
                clusters.append(
                    [
                        {
                            "embedding": emb,
                            "face": f,
                            "crop": crop_face(frame, f),
                            "score": float(
                                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                            ),
                        }
                    ]
                )

    characters = []
    for idx, cluster in enumerate(clusters):
        best = max(cluster, key=lambda x: x["score"])
        centroid = np.mean([x["embedding"] for x in cluster], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        thumb = img_to_b64(best["crop"])
        characters.append(
            {
                "characterId": idx,
                "thumbnailBase64": thumb,
                "embedding": centroid.tolist(),
            }
        )

    return characters


def build_assignments(swaps):
    assignments = []
    for s in swaps:
        src_b64 = s.get("newFace") or s.get("sourceImage")
        target_embedding = s.get("targetEmbedding")
        if not src_b64 or target_embedding is None:
            continue

        src_img = bytes_to_img(b64_to_bytes(src_b64))
        if src_img is None:
            continue

        src_faces = app.get(src_img)
        if not src_faces:
            continue

        largest = max(
            src_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        assignments.append(
            {
                "target_embedding": np.asarray(target_embedding, dtype=np.float32),
                "source_face": largest,
            }
        )

    return assignments


def find_matching_swap(face_embedding, assignments, threshold=MATCH_THRESH):
    best_src = None
    best_sim = -1.0
    for s in assignments:
        sim = cosine(face_embedding, s["target_embedding"])
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_src = s["source_face"]
    return best_src


def apply_enhancer(img, pipeline):
    if pipeline not in (None, "", "fast", "pipeline1", "1"):
        restorer = get_gfpgan()
        if restorer not in (None, False):
            try:
                _, _, out = restorer.enhance(img, paste_back=True)
                return out
            except Exception as e:
                print("GFPGAN failed:", e, flush=True)
    return img


def process_frame(frame, assignments, enhance=False, pipeline="fast"):
    faces = app.get(frame)
    if not faces:
        return frame

    out = frame.copy()
    for face in faces:
        src_face = find_matching_swap(face.embedding, assignments)
        if src_face is not None:
            out = swapper.get(out, face, src_face, paste_back=True)

    if enhance:
        out = apply_enhancer(out, pipeline)

    return out


def handle_detect(media_b64, isvideo):
    data = b64_to_bytes(media_b64)
    kind = sniff_kind(data, isvideo)

    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        frames = [img]

    elif kind == "gif":
        all_frames, _ = gif_frames(data)
        idxs = np.linspace(0, len(all_frames) - 1, min(DETECT_SAMPLES, len(all_frames))).astype(int)
        frames = [all_frames[i] for i in idxs]

    else:
        path = write_temp(data, ".mp4")
        try:
            frames = sample_video_frames(path, DETECT_SAMPLES)
        finally:
            os.remove(path)
        if not frames:
            return {"error": "Could not read video frames"}

    return {"characters": cluster_faces(frames), "mediaKind": kind}


def handle_swap(media_b64, isvideo, swaps, pipeline):
    assignments = build_assignments(swaps)
    if not assignments:
        return {"error": "No usable face found in the replacement photos"}

    enhance = pipeline not in (None, "", "fast", "pipeline1", "1")

    data = b64_to_bytes(media_b64)
    kind = sniff_kind(data, isvideo)

    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        img = process_frame(img, assignments, enhance, pipeline)
        return {"mediaType": "image", "base64": img_to_b64(img)}

    if kind == "gif":
        frames, durations = gif_frames(data)
        out = []
        for fr in frames:
            fr = process_frame(fr, assignments, enhance, pipeline)
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))

        buf = io.BytesIO()
        out[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=out[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )
        return {
            "mediaType": "gif",
            "base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        }

    inpath = write_temp(data, ".mp4")
    rawpath = inpath + ".raw.mp4"
    finalpath = inpath + ".final.mp4"

    try:
        cap = cv2.VideoCapture(inpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            rawpath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(process_frame(frame, assignments, enhance, pipeline))

        cap.release()
        writer.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                rawpath,
                "-i",
                inpath,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                finalpath,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with open(finalpath, "rb") as f:
            return {
                "mediaType": "video",
                "base64": base64.b64encode(f.read()).decode("utf-8"),
            }

    finally:
        for p in (inpath, rawpath, finalpath):
            if os.path.exists(p):
                os.remove(p)


def handler(job):
    job_input = job.get("input", {})

    action = job_input.get("action", "swap")
    media_b64 = job_input.get("media") or job_input.get("targetImage") or job_input.get("targetimage")
    isvideo = bool(job_input.get("isVideo", job_input.get("isvideo", False)))
    pipeline = str(job_input.get("pipeline", "fast")).lower()

    if not media_b64:
        return {
            "status": "failed",
            "error": f"Missing media payload. Keys received: {list(job_input.keys())}",
        }

    try:
        if action == "detect":
            result = handle_detect(media_b64, isvideo)
            if "error" in result:
                return {"status": "failed", "error": result["error"]}
            return {
                "status": "success",
                "characters": result["characters"],
                "mediaKind": result["mediaKind"],
            }

        if action == "swap":
            swaps = job_input.get("swaps", [])
            result = handle_swap(media_b64, isvideo, swaps, pipeline)
            if "error" in result:
                return {"status": "failed", "error": result["error"]}
            return {
                "status": "success",
                "mediaType": result["mediaType"],
                "base64": result["base64"],
            }

        return {"status": "failed", "error": f"Invalid action: {action}"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


runpod.serverless.start({"handler": handler})
