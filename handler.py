import os
import cv2
import base64
import tempfile
import urllib.request
import numpy as np
import runpod
import insightface
from insightface.app import FaceAnalysis

print("Worker: Initializing Puthiya Mugam handler...")

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

faceapp = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
faceapp.prepare(ctx_id=0, det_size=(640, 640))

def find_model_path():
    candidates = [
        os.environ.get("SWAPPERPATH"),
        os.environ.get("INSWAPPERPATH"),
        "/app/models/inswapper128.onnx",
        "/app/models/inswapper_128.onnx",
        "/models/inswapper128.onnx",
        "/models/inswapper_128.onnx",
        os.path.expanduser("~/.insightface/models/inswapper128.onnx"),
        os.path.expanduser("~/.insightface/models/inswapper_128.onnx"),
        "./inswapper128.onnx",
        "./inswapper_128.onnx",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None

SWAPPER_PATH = find_model_path()
swapper = None

print(f"Worker: Resolved SWAPPER_PATH to -> {SWAPPER_PATH}")

if SWAPPER_PATH:
    try:
        swapper = insightface.modelzoo.get_model(
            SWAPPER_PATH,
            download=False,
            providers=PROVIDERS
        )
        print(f"Worker: inswapper successfully loaded from {SWAPPER_PATH}")
    except Exception as e:
        print(f"Worker: failed to load inswapper: {e}")
else:
    print("Worker ERROR: inswapper model file not found in expected locations.")

gfpgan_enhancer = None
try:
    from gfpgan import GFPGANer
    gfpgan_path = os.environ.get("GFPGANPATH", "/app/models/GFPGANv1.4.pth")
    if os.path.exists(gfpgan_path):
        gfpgan_enhancer = GFPGANer(
            model_path=gfpgan_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None
        )
        print("Worker: GFPGAN loaded.")
except Exception as e:
    print(f"Worker: GFPGAN unavailable: {e}")

def strip_data_uri(value: str) -> str:
    if not value:
        return value
    if "," in value and value.strip().startswith("data:"):
        return value.split(",", 1)[1]
    return value.strip()

def resolve_media_to_bytes(media_str: str) -> bytes:
    if not media_str:
        return None
    media_str = media_str.strip()
    if media_str.startswith("http://") or media_str.startswith("https://"):
        req = urllib.request.Request(media_str, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    return base64.b64decode(strip_data_uri(media_str))

def decode_image_bytes(raw_bytes: bytes):
    if not raw_bytes:
        return None
    arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def extract_first_frame(media_bytes: bytes, is_video: bool):
    if not is_video:
        img = decode_image_bytes(media_bytes)
        if img is not None and getattr(img, "size", 0) > 0:
            return img

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def encode_jpg_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buf).decode("utf-8")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def find_matching_source(face_embedding, swaps_prepared, threshold=0.45):
    best_src = None
    best_sim = -1.0
    for s in swaps_prepared:
        sim = cosine_similarity(face_embedding, s["targetembedding"])
        if sim > threshold and sim > best_sim:
            best_sim = sim
            best_src = s["sourceface"]
    return best_src

def apply_gfpgan(img_bgr: np.ndarray):
    if gfpgan_enhancer is None:
        return img_bgr
    try:
        _, _, restored = gfpgan_enhancer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        return restored if restored is not None else img_bgr
    except Exception as e:
        print(f"Worker: GFPGAN failed: {e}")
        return img_bgr

def apply_enhancer(img_bgr: np.ndarray, pipeline: str) -> np.ndarray:
    pipeline = (pipeline or "fast").lower()
    if "gfpgan" in pipeline:
        return apply_gfpgan(img_bgr)
    return img_bgr

def prepare_swaps(swaps_input):
    swaps_prepared = []
    for s in swaps_input:
        new_face_raw = s.get("newface") or s.get("sourceimage")
        target_embedding = s.get("targetembedding")

        if not new_face_raw or target_embedding is None:
            continue

        try:
            src_bytes = resolve_media_to_bytes(new_face_raw)
            src_img = decode_image_bytes(src_bytes)
            if src_img is None:
                continue

            src_faces = faceapp.get(src_img)
            if not src_faces:
                continue

            swaps_prepared.append({
                "targetembedding": np.array(target_embedding, dtype=np.float32),
                "sourceface": src_faces[0]
            })
        except Exception as e:
            print(f"Worker: failed preparing swap source: {e}")

    return swaps_prepared

def detect_characters(frame):
    faces = faceapp.get(frame)
    characters = []
    h, w = frame.shape[:2]
    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        crop = frame[y1:y2, x1:x2]

        thumb_b64 = ""
        if crop is not None and crop.size > 0:
            thumb_b64 = encode_jpg_b64(crop)

        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
            if emb is not None:
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb

        if emb is None:
            continue

        characters.append({
            "index": idx,
            "thumbnailbase64": thumb_b64,
            "embedding": emb.tolist()
        })

    return characters

def swap_on_image(img, swaps_prepared, pipeline):
    if swapper is None:
        return {"error": "Inswapper model is not initialized or not found on worker."}

    faces = faceapp.get(img)
    out = img.copy()

    for face in faces:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
            if emb is not None:
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
        if emb is None:
            continue

        matched_source = find_matching_source(emb, swaps_prepared)
        if matched_source is not None:
            out = swapper.get(out, face, matched_source, paste_back=True)

    out = apply_enhancer(out, pipeline)
    return {
        "status": "success",
        "mediatype": "image",
        "base64": encode_jpg_b64(out)
    }

def swap_on_video(media_bytes, swaps_prepared, pipeline):
    if swapper is None:
        return {"error": "Inswapper model is not initialized or not found on worker."}

    in_path = None
    out_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(media_bytes)
            in_path = tmp.name

        out_path = in_path.replace(".mp4", "_out.mp4")

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            return {"error": "Invalid video dimensions."}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = faceapp.get(frame)
            swapped = frame.copy()

            for face in faces:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(face, "embedding", None)
                    if emb is not None:
                        norm = np.linalg.norm(emb)
                        emb = emb / norm if norm > 0 else emb
                if emb is None:
                    continue

                matched_source = find_matching_source(emb, swaps_prepared)
                if matched_source is not None:
                    swapped = swapper.get(swapped, face, matched_source, paste_back=True)

            swapped = apply_enhancer(swapped, pipeline)
            writer.write(swapped)

        cap.release()
        writer.release()

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "mediatype": "video",
            "base64": video_b64
        }

    except Exception as e:
        return {"error": f"Video swap failed: {str(e)}"}
    finally:
        for p in [in_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

def handler(event):
    try:
        jobinput = event.get("input", {}) or {}
        action = (jobinput.get("action") or "swap").lower()

        media_raw = jobinput.get("media") or jobinput.get("targetimage") or jobinput.get("image")
        if not media_raw:
            return {"error": "Missing media payload in request."}

        is_video = bool(jobinput.get("isvideo", False))
        pipeline = str(jobinput.get("pipeline", "fast"))

        try:
            media_bytes = resolve_media_to_bytes(media_raw)
        except Exception as e:
            return {"error": f"Failed reading media: {str(e)}"}

        if not media_bytes:
            return {"error": "Media buffer is empty."}

        if action == "detect":
            frame = extract_first_frame(media_bytes, is_video)
            if frame is None:
                return {"error": "Could not decode media to scan for faces."}
            characters = detect_characters(frame)
            return {"status": "success", "characters": characters}

        if action == "swap":
            swaps_input = jobinput.get("swaps", [])
            if not swaps_input:
                return {"error": "Missing swaps list."}

            swaps_prepared = prepare_swaps(swaps_input)
            if not swaps_prepared:
                return {"error": "Could not detect a valid face in any uploaded new face image."}

            if not is_video:
                img = decode_image_bytes(media_bytes)
                if img is None:
                    return {"error": "Failed decoding target image."}
                return swap_on_image(img, swaps_prepared, pipeline)

            return swap_on_video(media_bytes, swaps_prepared, pipeline)

        if action == "info":
            return {
                "status": "active",
                "swapper_loaded": swapper is not None,
                "swapper_path": SWAPPER_PATH
            }

        return {"error": f"Unknown action: {action}"}

    except Exception as e:
        return {"error": f"Unhandled worker error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
