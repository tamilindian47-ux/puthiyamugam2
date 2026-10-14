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

print("=== Starting Worker Initialization ===", flush=True)

try:
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
    
    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l", root="/root/.insightface",
        allowed_modules=['detection', 'recognition'], providers=providers
    )
    face_app.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        "/app/checkpoints/inswapper_128.onnx", download=False, providers=providers
    )

    print("=== Worker Successfully Initialized ===", flush=True)
except Exception as e:
    traceback.print_exc()
    raise e

def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")

    if not media_data:
        return {"error": "Missing 'media' payload."}

    tmp_media = f"/tmp/input_{uuid.uuid4().hex[:8]}.jpg"
    try:
        if media_data.startswith("http"):
            resp = requests.get(media_data, timeout=60)
            with open(tmp_media, "wb") as f:
                f.write(resp.content)
        else:
            if "," in media_data:
                media_data = media_data.split(",", 1)[1]
            with open(tmp_media, "wb") as f:
                f.write(base64.b64decode(media_data))

        frame = cv2.imread(tmp_media)
        faces = face_app.get(frame)
        return {"status": "success", "faces_detected": len(faces)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)

runpod.serverless.start({"handler": handler})
