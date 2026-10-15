import os
import sys
import cv2
import uuid
import queue
import base64
import threading
import traceback
import subprocess
import requests
import numpy as np
import onnxruntime as ort
import insightface
import runpod

CRF = os.environ.get("OUTPUT_CRF", "18")
PRESET = os.environ.get("OUTPUT_PRESET", "p4")
ENHANCE_BLEND = float(os.environ.get("ENHANCE_BLEND", "0.25"))

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
    try:
        ort.preload_dlls()
    except Exception as preload_err:
        print(f"preload_dlls warning: {preload_err}")

    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers detected: {available_providers}")
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(f"FATAL: CUDAExecutionProvider not found! Detected: {available_providers}")

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = [("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": 0,
        "arena_extend_strategy": "kNextPowerOfTwo"
    })]

    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root="/root/.insightface",
        allowed_modules=["detection", "recognition"],
        providers=providers
    )
    face_app.prepare(ctx_id=0, det_thresh=0.25, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        "/app/checkpoints/inswapper_128.onnx",
        download=False,
        providers=providers
    )

    occ_session = ort.InferenceSession("/app/checkpoints/face_occluder.onnx", sess_options=session_opts, providers=providers)
    enh_session = ort.InferenceSession("/app/checkpoints/enhancer.onnx", sess_options=session_opts, providers=providers)

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

    _BOX_MASK_CACHE = {}

    def get_box_mask(size=128, blur=0.35):
        key = (size, blur)
        if key not in _BOX_MASK_CACHE:
            amount = int(size * 0.35 * blur)
            edge = max(amount // 2, 1)
            m = np.ones((size, size), np.float32)
            m[:edge, :] = 0
            m[-edge:, :] = 0
            m[:, :edge] = 0
            m[:, -edge:] = 0
            _BOX_MASK_CACHE[key] = cv2.GaussianBlur(m, (0, 0), max(amount * 0.25, 0.1))
        return _BOX_MASK_CACHE[key]

    print("=== Worker Successfully Initialized: GPU Accelerators Bound ===")
    sys.stdout.flush()

except Exception as init_err:
    print("FATAL INITIALIZATION ERROR:")
    traceback.print_exc()
    sys.stdout.flush()
    raise init_err


def occlusion_mask(crop_bgr):
    img = cv2.resize(crop_bgr, occ_size).astype(np.float32) / 255.0
    if occ_nchw:
        img = img.transpose(2, 0, 1)
    out = occ_session.run(None, {occ_input.name: img[None]})[0][0]
    m = np.clip(np.squeeze(out), 0, 1).astype(np.float32)
    m = cv2.resize(m, (crop_bgr.shape[1], crop_bgr.shape[0]))
    m = (np.clip(cv2.GaussianBlur(m, (0, 0), 2.5), 0.3, 1) - 0.3) * 1.42
    return np.clip(m, 0, 1)


def swap_face(frame, face, source_face, smooth_M=None):
    bgr_fake, M = swapper.get(frame, face, source_face, paste_back=False)
    if bgr_fake is None:
        return frame

    if smooth_M is not None:
        M = 0.75 * M + 0.25 * smooth_M

    bgr_fake = np.ascontiguousarray(bgr_fake)
    h, w = frame.shape[:2]
    crop = cv2.warpAffine(frame, M, (128, 128), borderValue=0)
    
    mask = np.minimum(get_box_mask(128, 0.35), occlusion_mask(crop))
    IM = cv2.invertAffineTransform(M)
    
    fake_w = cv2.warpAffine(bgr_fake, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    mask_w = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
    
    out = frame.astype(np.float32) * (1.0 - mask_w) + fake_w * mask_w
    return np.clip(out, 0, 255).astype(np.uint8)


def run_enhancer(crop_bgr):
    x = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    y = enh_session.run(None, {enh_input_name: x})[0][0]
    y = (np.clip(y, -1, 1) + 1.0) / 2.0
    y = np.clip(np.rint(y.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(y[:, :, ::-1])


def enhance_face(swapped, original, face, blend=ENHANCE_BLEND):
    if blend <= 0.05:
        return swapped
    kps = face.kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(kps, FFHQ_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return swapped
    h, w = swapped.shape[:2]
    s = enh_size
    crop = cv2.warpAffine(swapped, M, (s, s), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
    orig_crop = cv2.warpAffine(original, M, (s, s), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)

    enhanced = run_enhancer(crop)
    enhanced = crop.astype(np.float32) * (1.0 - blend) + enhanced.astype(np.float32) * blend
    mask = np.minimum(get_box_mask(s, 0.2), occlusion_mask(orig_crop))
    IM = cv2.invertAffineTransform(M)
    enh_w = cv2.warpAffine(enhanced, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask_w = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
    out = swapped.astype(np.float32) * (1.0 - mask_w) + enh_w.astype(np.float32) * mask_w
    return np.clip(out, 0, 255).astype(np.uint8)


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
        vec = d["embedding"]
        matched = False
        for c in clusters:
            centroid = np.mean([detections[idx]["embedding"] for idx in c], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
            if float(np.dot(vec, centroid)) >= (1.0 - threshold):
                c.append(i)
                matched = True
                break
        if not matched:
            clusters.append([i])

    unique_characters = []
    for l, c_indices in enumerate(clusters):
        best_idx = max(c_indices, key=lambda idx: detections[idx]["score"])
        centroid = np.mean([detections[idx]["embedding"] for idx in c_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
        _, buf = cv2.imencode(".jpg", detections[best_idx]["crop"])
        unique_characters.append({
            "character_id": int(l),
            "thumbnail_base64": base64.b64encode(buf).decode("utf-8"),
            "embedding": centroid.tolist()
        })
    return unique_characters


def collect_faces(frame, detections, min_size=0):
    for f in face_app.get(frame):
        box = f.bbox.astype(int)
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
        if (x2 - x1) > min_size and (y2 - y1) > min_size:
            detections.append({
                "crop": frame[y1:y2, x1:x2],
                "embedding": f.normed_embedding,
                "score": (x2 - x1) * (y2 - y1)
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


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / float(union + 1e-6)


def execute_swap(media_path, media_type, swap_targets, pipeline="fast"):
    enhance = pipeline in ("gfpgan", "codeformer", "enhanced")
    active_blend = ENHANCE_BLEND if enhance else 0.0

    if media_type == "image":
        frame = cv2.imread(media_path)
        faces = face_app.get(frame)
        for face in faces:
            for target in swap_targets:
                if float(np.dot(face.normed_embedding, target["target_embedding"])) >= 0.38:
                    original = frame.copy()
                    frame = swap_face(frame, face, target["source_face"])
                    if enhance:
                        frame = enhance_face(frame, original, face, blend=active_blend)
                    break
        _, out_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return {"status": "success", "media_type": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    temp_id = uuid.uuid4().hex[:8]
    tmp_mp4 = f"/tmp/final_{temp_id}.mp4"
    tmp_gif = f"/tmp/final_{temp_id}.gif"

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = width - (width % 2), height - (height % 2)

    # Use h264_nvenc with fallback to libx264
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "-",
        "-i", media_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        tmp_mp4
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except Exception:
        cmd[cmd.index("h264_nvenc")] = "libx264"
        cmd[cmd.index("-tune")] = "-crf"
        cmd[cmd.index("ll")] = CRF
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_q = queue.Queue(maxsize=16)
    write_q = queue.Queue(maxsize=16)

    def reader_worker():
        while True:
            ret, f = cap.read()
            if not ret:
                frame_q.put(None)
                break
            frame_q.put(f)

    def writer_worker():
        while True:
            item = write_q.get()
            if item is None:
                break
            proc.stdin.write(item)
            write_q.task_done()

    t_read = threading.Thread(target=reader_worker, daemon=True)
    t_write = threading.Thread(target=writer_worker, daemon=True)
    t_read.start()
    t_write.start()

    face_tracks = {}
    frame_idx = 0

    try:
        while True:
            f = frame_q.get()
            if f is None:
                break

            faces = face_app.get(f)
            new_tracks = {}

            for face in faces:
                matched_target = None
                matched_M = None
                best_iou = 0.0
                best_track_id = None

                for track_id, track in face_tracks.items():
                    iou = compute_iou(face.bbox, track["bbox"])
                    if iou > 0.45 and iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

                if best_track_id is not None:
                    matched_target = face_tracks[best_track_id]["target"]
                    matched_M = face_tracks[best_track_id].get("last_M")
                else:
                    for target in swap_targets:
                        sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
                        if sim >= 0.38:
                            matched_target = target
                            break

                if matched_target:
                    original = f.copy()
                    f = swap_face(f, face, matched_target["source_face"], smooth_M=matched_M)
                    if enhance:
                        f = enhance_face(f, original, face, blend=active_blend)

                    tid = best_track_id if best_track_id is not None else len(new_tracks)
                    new_tracks[tid] = {
                        "bbox": face.bbox,
                        "target": matched_target,
                        "last_M": getattr(face, "last_M", None)
                    }

            face_tracks = new_tracks
            frame_idx += 1

            if f.shape[1] != w2 or f.shape[0] != h2:
                f = f[:h2, :w2]

            write_q.put(np.ascontiguousarray(f).tobytes())

    finally:
        cap.release()
        write_q.put(None)
        t_write.join()
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg pipeline exited with code {rc}")

    final_path = tmp_mp4
    if media_type == "gif":
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_mp4,
            "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
            "-loop", "0", tmp_gif
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        final_path = tmp_gif

    with open(final_path, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode("utf-8")

    for fpath in [tmp_mp4, tmp_gif]:
        if os.path.exists(fpath):
            os.remove(fpath)

    return {"status": "success", "media_type": media_type, "base64": media_b64}


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")
    pipeline = job_input.get("pipeline", "fast")

    if not media_data:
        return {"error": "Missing 'media' payload."}

    if job_input.get("is_gif") or (isinstance(media_data, str) and ".gif" in media_data.lower()[:100]):
        media_type, ext = "gif", "gif"
    elif job_input.get("is_video", True):
        media_type, ext = "video", "mp4"
    else:
        media_type, ext = "image", "jpg"

    tmp_media = f"/tmp/input_{uuid.uuid4().hex[:8]}.{ext}"

    try:
        resolve_input_to_file(media_data, tmp_media)

        if action == "detect":
            characters = detect_unique_people(tmp_media, is_animation=(media_type in ["video", "gif"]))
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
            return execute_swap(tmp_media, media_type, swap_targets, pipeline)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)


runpod.serverless.start({"handler": handler})
