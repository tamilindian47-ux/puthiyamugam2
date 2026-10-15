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

CRF = os.environ.get("OUTPUT_CRF", "18")
PRESET = os.environ.get("OUTPUT_PRESET", "fast")
ENHANCE_BLEND = float(os.environ.get("ENHANCE_BLEND", "0.25"))

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
    try:
        ort.preload_dlls()
    except Exception as preload_err:
        print(f"preload_dlls warning: {preload_err}")

    providers = [("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC"
    })]

    # Full app: used only for 1-time scanning and source image embedding extraction
    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root="/root/.insightface",
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    face_app.prepare(ctx_id=0, det_thresh=0.30, det_size=(640, 640))

    # Fast detection-only handle: bypasses ArcFace recognition on video frames
    det_model = face_app.models['detection']

    swapper = insightface.model_zoo.get_model(
        "/app/checkpoints/inswapper_128.onnx",
        download=False,
        providers=providers
    )

    occ_session = ort.InferenceSession("/app/checkpoints/face_occluder.onnx", providers=providers)
    enh_session = ort.InferenceSession("/app/checkpoints/enhancer.onnx", providers=providers)

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

    _BOX_MASK_128 = None
    def get_box_mask_128():
        global _BOX_MASK_128
        if _BOX_MASK_128 is None:
            m = np.ones((128, 128), np.float32)
            m[:9, :] = 0; m[-9:, :] = 0; m[:, :9] = 0; m[:, -9:] = 0
            _BOX_MASK_128 = cv2.GaussianBlur(m, (0, 0), 4.0)
        return _BOX_MASK_128

    print("=== Worker Successfully Initialized ===")
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
    m = (np.clip(cv2.GaussianBlur(m, (0, 0), 2.5), 0.35, 1) - 0.35) * 1.53
    return np.clip(m, 0, 1)


def swap_face(frame, face_kps, source_face):
    # Construct minimal face object required by inswapper
    class FaceProxy:
        def __init__(self, kps):
            self.kps = kps
            self.embedding = None

    proxy = FaceProxy(face_kps)
    bgr_fake, M = swapper.get(frame, proxy, source_face, paste_back=False)
    if bgr_fake is None:
        return frame

    bgr_fake = np.ascontiguousarray(bgr_fake)
    h, w = frame.shape[:2]
    crop = cv2.warpAffine(frame, M, (128, 128), borderValue=0)
    
    mask = np.minimum(get_box_mask_128(), occlusion_mask(crop))
    IM = cv2.invertAffineTransform(M)
    
    fake_w = cv2.warpAffine(bgr_fake, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    mask_w = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
    
    out = frame.astype(np.float32) * (1.0 - mask_w) + fake_w * mask_w
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
        vec = d['embedding']
        matched = False
        for c in clusters:
            centroid = np.mean([detections[idx]['embedding'] for idx in c], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
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
        centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
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
        step = max(1, total_frames // 25)
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
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def execute_swap(media_path, media_type, swap_targets, pipeline="fast"):
    if media_type == "image":
        frame = cv2.imread(media_path)
        faces = face_app.get(frame)
        for face in faces:
            for target in swap_targets:
                sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
                if sim >= 0.38:
                    frame = swap_face(frame, face.kps, target["source_face"])
                    break
        _, out_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return {"status": "success", "media_type": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    temp_id = uuid.uuid4().hex[:8]
    tmp_mp4 = f"/tmp/final_{temp_id}.mp4"

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = width - (width % 2), height - (height % 2)

    # Use hardware NVENC encoder, fallback to fast libx264
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

    single_target = swap_targets[0]["source_face"] if len(swap_targets) == 1 else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Fast detection without ArcFace recognition
            bboxes, kpss = det_model.detect(frame, max_num=0, metric='default')
            
            if bboxes is not None and len(bboxes) > 0:
                for idx, kps in enumerate(kpss):
                    if single_target:
                        frame = swap_face(frame, kps, single_target)
                    else:
                        # Fallback to full recognition only if multiple target identities exist
                        faces = face_app.get(frame)
                        for f in faces:
                            for target in swap_targets:
                                if float(np.dot(f.normed_embedding, target["target_embedding"])) >= 0.38:
                                    frame = swap_face(frame, f.kps, target["source_face"])
                                    break
                        break

            if frame.shape[1] != w2 or frame.shape[0] != h2:
                frame = frame[:h2, :w2]

            proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg pipeline exited with code {rc}")

    with open(tmp_mp4, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode("utf-8")

    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)

    return {"status": "success", "media_type": "video", "base64": media_b64}


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")
    pipeline = job_input.get("pipeline", "fast")

    if not media_data:
        return {"error": "Missing 'media' payload."}

    media_type = "image" if not job_input.get("is_video", True) else "video"
    ext = "jpg" if media_type == "image" else "mp4"
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
            return execute_swap(tmp_media, media_type, swap_targets, pipeline)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)


runpod.serverless.start({"handler": handler})
