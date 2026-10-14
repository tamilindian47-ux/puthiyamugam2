import sys
# Compatibility patch for Python 3.10 running FaceFusion's Python 3.11+ typing requirements
try:
    import typing_extensions
    if not hasattr(sys.modules.get('typing', typing), 'NotRequired'):
        import typing
        typing.NotRequired = typing_extensions.NotRequired
        typing.Required = typing_extensions.Required
except Exception:
    pass

import os
import subprocess
import base64
import tempfile
import runpod
import cv2
import numpy as np

print("Worker: Initializing FaceFusion Serverless Handler...")

_faceapp = None
def get_faceapp():
    global _faceapp
    if _faceapp is None:
        import insightface
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _faceapp = FaceAnalysis(name="buffalo_l", providers=providers)
        _faceapp.prepare(ctx_id=0, det_size=(640, 640))
    return _faceapp

def resolve_media_to_bytes(media_str: str) -> bytes:
    if not media_str:
        return None
    media_str = media_str.strip()
    if media_str.startswith("http://") or media_str.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(media_str, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    if "," in media_str and media_str.startswith("data:"):
        media_str = media_str.split(",", 1)[1]
    return base64.b64decode(media_str)

def decode_image_bytes(raw_bytes: bytes):
    if not raw_bytes:
        return None
    arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def extract_first_frame(media_bytes: bytes, is_video: bool):
    if not is_video:
        return decode_image_bytes(media_bytes)
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

def encode_jpg_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Failed to encode thumbnail image.")
    return base64.b64encode(buf).decode("utf-8")

def detect_characters(frame):
    app = get_faceapp()
    faces = app.get(frame)
    characters = []
    h, w = frame.shape[:2]
    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        crop = frame[y1:y2, x1:x2]
        
        thumb_b64 = encode_jpg_b64(crop) if crop is not None and crop.size > 0 else ""
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

def handler(event):
    try:
        jobinput = event.get("input", {}) or {}
        action = (jobinput.get("action") or "swap").lower()
        
        media_raw = jobinput.get("media") or jobinput.get("targetimage") or jobinput.get("image")
        if not media_raw:
            return {"error": "Missing media payload in request."}

        is_video = bool(jobinput.get("isvideo", False))
        media_bytes = resolve_media_to_bytes(media_raw)
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

            target_suffix = ".mp4" if is_video else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=target_suffix) as tmp_target:
                tmp_target.write(media_bytes)
                target_path = tmp_target.name

            output_path = target_path.replace(target_suffix, f"_out{target_suffix}")
            
            source_path = target_path.replace(target_suffix, "_source.jpg")
            source_bytes = resolve_media_to_bytes(swaps_input[0].get("newface"))
            with open(source_path, "wb") as f:
                f.write(source_bytes)

            # Map user parameters (defaulting to HyperSwap 1A for extreme angles/motion)
            swapper_model = jobinput.get("model", "hyperswap_1a_256")
            enhancer = jobinput.get("enhancer", "codeformer")
            enhancer_blend = str(jobinput.get("enhancer_blend", 35))
            expression_factor = str(jobinput.get("expression_factor", 85))

            processors = ["face_swapper", "expression_restorer"]
            if enhancer != "none":
                processors.append("face_enhancer")

            cmd = [
                "python", "facefusion.py", "headless-run",
                "--target", target_path,
                "--source", source_path,
                "--output", output_path,
                "--processors", *processors,
                "--face-swapper-model", swapper_model,
                "--face-enhancer-model", enhancer if enhancer != "none" else "none",
                "--face-enhancer-blend", enhancer_blend,
                "--expression-restorer-model", "live_portrait",
                "--expression-restorer-factor", expression_factor,
                "--execution-providers", "cuda"
            ]

            print(f"Executing FaceFusion CLI: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app/facefusion")
            
            if result.returncode != 0:
                return {"error": f"FaceFusion execution failed: {result.stderr.strip()}"}

            if not os.path.exists(output_path):
                return {"error": "FaceFusion finished but output file was not generated."}

            with open(output_path, "rb") as f:
                output_bytes = f.read()

            for p in [target_path, source_path, output_path]:
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except: pass

            return {
                "status": "success",
                "mediatype": "video" if is_video else "image",
                "base64": base64.b64encode(output_bytes).decode("utf-8")
            }

        return {"error": f"Unknown action: {action}"}

    except Exception as e:
        return {"error": f"Unhandled worker error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
