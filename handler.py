import os
import sys
import cv2
import uuid
import base64
import traceback
import subprocess
import urllib.request
import numpy as np
import onnxruntime as ort
import insightface
import runpod

CRF = os.environ.get("OUTPUT_CRF", "17")
PRESET = os.environ.get("OUTPUT_PRESET", "fast")
ENHANCE_BLEND = float(os.environ.get("ENHANCE_BLEND", "0.7"))

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
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
        "/app/models/inswapper128.onnx", download=False, providers=providers
    )
    occ_session = ort.InferenceSession("/app/models/face_occluder.onnx", providers=providers)
    enh_session = ort.InferenceSession("/app/models/enhancer.onnx", providers=providers)

    print("=== Worker Successfully Initialized: Models Residing in VRAM ===")
    sys.stdout.flush()

except Exception as init_err:
    print("FATAL INITIALIZATION ERROR:")
    traceback.print_exc()
    sys.stdout.flush()
    raise init_err

occ_input = occ_session.get_inputs()[0]
occ_shape = occ_input.shape
occ_nchw = (len(occ_shape) == 4 and occ_shape[1] == 3)
try:
    occ_size = (int(occ_shape[3]), int(occ_shape[2])) if occ_nchw else (int(occ_shape[2]), int(occ_shape[1]))
except Exception:
    occ_size = (256, 256)

enh_input_name = enh_session.get_inputs()[0].name
enh_size = int(enh_session.get_inputs()[0].shape[2])
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
    [0.39308822, 0.72541100], [0.61150205, 0.72490465],
], dtype=np.float32) * enh_size


def occlusion_mask(crop_bgr):
    img = cv2.resize(crop_bgr, occ_size).astype(np.float32) / 255.0
    if occ_nchw:
        img = img.transpose(2, 0, 1)
    out = occ_session.run(None, {occ_input.name: img[None]})[0][0]
    m = np.clip(np.squeeze(out), 0, 1).astype(np.float32)
    m = cv2.resize(m, (crop_bgr.shape[1], crop_bgr.shape[0]))
    m = (np.clip(cv2.GaussianBlur(m, (0, 0), 5), 0.5, 1) - 0.5) * 2
    return m


def box_mask(size=128, blur=0.5):
    amount = int(size * 0.5 * blur)
    edge = max(amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:edge, :] = 0
    m[-edge:, :] = 0
    m[:, :edge] = 0
    m[:, -edge:] = 0
    return cv2.GaussianBlur(m, (0, 0), max(amount * 0.25, 0.1))


def swap_face(frame, face, source_face):
    bgr_fake, M = swapper.get(frame, face, source_face, paste_back=False)
    bgr_fake = np.ascontiguousarray(bgr_fake)
    h, w = frame.shape[:2]
    crop = cv2.warpAffine(frame, M, (128, 128), borderValue=0)
    mask = np.minimum(box_mask(), occlusion_mask(crop))
    IM = cv2.invertAffineTransform(M)
    fake_w = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0).astype(np.float32)
    mask_w = cv2.warpAffine(mask, IM, (w, h), borderValue=0)[:, :, None]
    out = frame.astype(np.float32) * (1 - mask_w) + fake_w * mask_w
    return np.clip(out, 0, 255).astype(np.uint8)


def run_enhancer(crop_bgr):
    x = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    y = enh_session.run(None, {enh_input_name: x})[0][0]
    y = (np.clip(y, -1, 1) + 1) / 2
    y = np.clip(np.rint(y.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(y[:, :, ::-1])


def enhance_face(swapped, original, face, blend=ENHANCE_BLEND):
    kps = face.kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(kps, FFHQ_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return swapped
    h, w = swapped.shape[:2]
    s = enh_size
    crop = cv2.warpAffine(swapped, M, (s, s), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    orig_crop = cv2.warpAffine(original, M, (s, s), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    enhanced = run_enhancer(crop)
    enhanced = crop.astype(np.float32) * (1 - blend) + enhanced.astype(np.float32) * blend
    mask = np.minimum(box_mask(s, 0.3), occlusion_mask(orig_crop))
    IM = cv2.invertAffineTransform(M)
    enh_w = cv2.warpAffine(enhanced, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask_w = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
    out = swapped.astype(np.float32) * (1 - mask_w) + enh_w.astype(np.float32) * mask_w
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_media_to_bytes(media_str: str) -> bytes:
    if not media_str:
        return None
    media_str = media_str.strip()
    if media_str.startswith("http://") or media_str.startswith("https://"):
        req = urllib.request.Request(media_str, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    if "," in media_str and media_str.startswith("data:"):
        media_str = media_str.split(",", 1)[1]
    return base64.b64decode(media_str)


def decode_image(data_or_url):
    temp = f"/tmp/img_{uuid.uuid4().hex[:6]}.jpg"
    try:
        with open(temp, "wb") as f:
            f.write(resolve_media_to_bytes(data_or_url))
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
        centroid = np.mean([detections[idx]['embedding'] for idx in c_indices], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
        unique_characters.append({
            "index": int(l),
            "thumbnailbase64": base64.b64encode(buf).decode("utf-8"),
            "embedding": centroid.tolist()
        })
    return unique_characters


def detect_characters_from_media(media_bytes: bytes, is_video: bool):
    detections = []
    tmp_path = None
    try:
        ext = ".mp4" if is_video else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name
            
        if not is_video:
            frame = cv2.imread(tmp_path)
            if frame is not None:
                for f in face_app.get(frame):
                    box = f.bbox.astype(int)
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'crop': frame[y1:y2, x1:x2],
                            'embedding': f.normed_embedding,
                            'score': (x2 - x1) * (y2 - y1)
                        })
        else:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            step = max(1, total_frames // 30)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    for f in face_app.get(frame):
                        box = f.bbox.astype(int)
                        x1, y1 = max(0, box[0]), max(0, box[1])
                        x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                        if (x2 - x1) > 40 and (y2 - y1) > 40:
                            detections.append({
                                'crop': frame[y1:y2, x1:x2],
                                'embedding': f.normed_embedding,
                                'score': (x2 - x1) * (y2 - y1)
                            })
                frame_idx += 1
            cap.release()

        if not detections:
            return []
        return cluster_embeddings_numpy(detections)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass


def apply_swaps(frame, swap_targets, enhance=False):
    for face in face_app.get(frame):
        for target in swap_targets:
            sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
            if sim >= 0.40:
                original = frame
                frame = swap_face(frame, face, target["source_face"])
                if enhance:
                    frame = enhance_face(frame, original, face)
                break
    return frame


def execute_swap(media_bytes, is_video, swap_targets, pipeline="codeformer"):
    enhance = pipeline != "none"
    print(f"Pipeline: {pipeline} | enhance={enhance} | CRF={CRF} preset={PRESET}")

    temp_id = uuid.uuid4().hex[:8]
    ext = ".mp4" if is_video else ".jpg"
    tmp_input = f"/tmp/input_{temp_id}{ext}"
    with open(tmp_input, "wb") as f:
        f.write(media_bytes)

    if not is_video:
        frame = apply_swaps(cv2.imread(tmp_input), swap_targets, enhance)
        _, out_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        if os.path.exists(tmp_input): os.remove(tmp_input)
        return {"status": "success", "mediatype": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    tmp_mp4 = f"/tmp/final_{temp_id}.mp4"
    cap = cv2.VideoCapture(tmp_input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame of video")
    h, w = first.shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "-",
        "-i", tmp_input,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", PRESET, "-crf", CRF, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        tmp_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame = first
    try:
        while True:
            out = apply_swaps(frame, swap_targets, enhance)
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

    for fpath in [tmp_input, tmp_mp4]:
        if os.path.exists(fpath):
            os.remove(fpath)

    return {"status": "success", "mediatype": "video", "base64": media_b64}


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
            characters = detect_characters_from_media(media_bytes, is_video)
            return {"status": "success", "characters": characters}

        if action == "swap":
            swaps_input = jobinput.get("swaps", [])
            if not swaps_input:
                return {"error": "Missing swaps list."}

            pipeline = jobinput.get("enhancer", "codeformer")
            swap_targets = []
            for s in swaps_input:
                src_faces = face_app.get(decode_image(s["newface"]))
                if src_faces:
                    swap_targets.append({
                        "target_embedding": np.array(s["targetembedding"], dtype=np.float32),
                        "source_face": src_faces[0]
                    })
            return execute_swap(media_bytes, is_video, swap_targets, pipeline)

        return {"error": f"Unknown action: {action}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Unhandled worker error: {str(e)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
