import os
import sys
import cv2
import uuid
import base64
import traceback
import subprocess
import requests
import numpy as np
import onnxruntime as ort
import insightface
import runpod

CRF = os.environ.get("OUTPUT_CRF", "17")
PRESET = os.environ.get("OUTPUT_PRESET", "fast")

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
    try:
        ort.preload_dlls()
    except Exception as preload_err:
        print(f"preload_dlls warning: {preload_err}")

    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers detected: {available_providers}")
    if 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError(f"FATAL: CUDAExecutionProvider not found! Detected: {available_providers}")

    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]

    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l", root="/root/.insightface",
        allowed_modules=['detection', 'recognition'], providers=providers
    )
    face_app.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        "/app/checkpoints/inswapper_128.onnx", download=False, providers=providers
    )

    for name, m in face_app.models.items():
        active = m.session.get_providers()
        print(f"Model '{name}' active providers: {active}")
        if 'CUDAExecutionProvider' not in active:
            raise RuntimeError(f"FATAL: '{name}' is NOT on GPU (providers: {active})")

    print("=== Worker Successfully Initialized: Models Residing in VRAM ===")
    sys.stdout.flush()

except Exception as init_err:
    print("FATAL INITIALIZATION ERROR:")
    traceback.print_exc()
    sys.stdout.flush()
    raise init_err


def swap_face(frame, face, source_face):
    # Perform direct face swap using inswapper
    bgr_fake, M = swapper.get(frame, face, source_face, paste_back=True)
    return np.clip(bgr_fake, 0, 255).astype(np.uint8)


def resolve_input_to_file(data_or_url, out_path):
    if data_or_url.startswith("http://") or data_or_url.startswith("https://"):
        resp = requests.get(data_or_url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        if "," in data_or_url:
            data_or_url = data_or_url.split(",", 1)[1]
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(data_or_url))


def decode_image(data_or_url):
    temp = f"/tmp/img_{uuid.uuid4().hex[:6]}.jpg"
    try:
        resolve_input_to_file(data_or_url, temp)
        return cv2.imread(temp)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def cluster_embeddings_numpy(detections, threshold=0.42):
    clusters = []
    for i, d in enumerate(detections):
        vec = d['embedding']
        matched = False
        for c in clusters:
            centroid = np.mean([detections[idx]['embedding'] for idx in c], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            if float(np.dot(vec, centroid)) >= (1.0 - threshold):
                c.append(i)
                matched = True
                break
        if not matched:
            clusters.append([i])

    unique_characters = []
    for l, c_indices in enumerate(clusters):
        best_idx = max(c_indices, key=lambda idx: detections[idx]['score'])
        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
        unique_characters.append({
            "character_id": int(l),
            "thumbnail_base64": base64.b64encode(buf).decode("utf-8"),
            "embedding": np.mean([detections[idx]['embedding'] for idx in c_indices], axis=0).tolist()
        })
    return unique_characters


def collect_faces(frame, detections, min_size=0):
    for f in face_app.get(frame):
        box = f.bbox.astype(int)
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
        if (x2 - x1) > min_size and (y2 - y1) > min_size:
            detections.append({
                'crop': frame[y1:y2, x1:x2],
                'embedding': f.normed_embedding,
                'score': (x2 - x1) * (y2 - y1)
            })


def detect_unique_people(media_path, is_animation=True):
    detections = []
    if not is_animation:
        collect_faces(cv2.imread(media_path), detections)
    else:
        cap = cv2.VideoCapture(media_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        step = max(1, total_frames // 30)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                collect_faces(frame, detections, min_size=40)
            frame_idx += 1
        cap.release()
    if not detections:
        return []
    return cluster_embeddings_numpy(detections)


def apply_swaps(frame, swap_targets):
    for face in face_app.get(frame):
        for target in swap_targets:
            sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
            if sim >= 0.40:
                frame = swap_face(frame, face, target["source_face"])
                break
    return frame


def execute_swap(media_path, media_type, swap_targets):
    if media_type == "image":
        frame = apply_swaps(cv2.imread(media_path), swap_targets)
        _, out_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return {"status": "success", "media_type": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    temp_id = uuid.uuid4().hex[:8]
    tmp_mp4 = f"/tmp/final_{temp_id}.mp4"

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame of video")
    h, w = first.shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)

    # Stream frames into FFmpeg via subprocess pipe
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "-",
        "-i", media_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", PRESET, "-crf", CRF, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        tmp_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame = first
    try:
        while True:
            out = apply_swaps(frame, swap_targets)
            if out.shape[1] != w2 or out.shape[0] != h2:
                out = out[:h2, :w2]
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            ret, frame = cap.read()
            if not ret:
                break
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed with code {rc}")

    with open(tmp_mp4, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode("utf-8")

    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)

    return {"status": "success", "media_type": media_type, "base64": media_b64}


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")

    if not media_data:
        return {"error": "Missing 'media' payload."}

    media_type = "video" if job_input.get("is_video", True) else "image"
    ext = "mp4" if media_type == "video" else "jpg"
    tmp_media = f"/tmp/input_{uuid.uuid4().hex[:8]}.{ext}"

    try:
        resolve_input_to_file(media_data, tmp_media)

        if action == "detect":
            characters = detect_unique_people(tmp_media, is_animation=(media_type == "video"))
            return {"status": "success", "characters": characters}

        elif action == "swap":
            swap_targets = []
            for s in job_input.get("swaps", []):
                src_faces = face_app.get(decode_image(s["new_face"]))
                if src_faces:
                    swap_targets.append({
                        "target_embedding": np.array(s["target_embedding"], dtype=np.float32),
                        "source_face": src_faces[0]
                    })
            return execute_swap(tmp_media, media_type, swap_targets)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)


runpod.serverless.start({"handler": handler})
