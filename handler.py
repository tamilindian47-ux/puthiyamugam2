import os
import subprocess
import base64
import tempfile
import runpod
from fastapi import FastAPI

print("Worker: Initializing FaceFusion Native Handler...")

def handler(event):
    jobinput = event.get("input", {}) or {}
    action = (jobinput.get("action") or "swap").lower()
    
    media_raw = jobinput.get("media") or jobinput.get("targetimage") or jobinput.get("image")
    if not media_raw:
        return {"error": "Missing media payload."}

    is_video = bool(jobinput.get("isvideo", False))
    
    # Map UI selections to FaceFusion parameters
    swapper_model = jobinput.get("model", "hyperswap_1c") # hyperswap_1c, hyperswap_1a, inswapper_128
    enhancer = jobinput.get("enhancer", "codeformer") # codeformer, gpen, none
    enhancer_blend = jobinput.get("enhancer_blend", 35) # Conservative 35% blend to protect micro-expressions
    expression_factor = jobinput.get("expression_factor", 80) # LivePortrait expression recovery factor

    # Decode target media
    target_suffix = ".mp4" if is_video else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=target_suffix) as tmp_target:
        if "," in media_raw:
            media_raw = media_raw.split(",")[1]
        tmp_target.write(base64.b64decode(media_raw))
        target_path = tmp_target.name

    output_path = target_path.replace(target_suffix, f"_out{target_suffix}")

    try:
        if action == "detect":
            # FaceFusion detection/indexing logic can be invoked or lightweight-scanned here
            # Returning basic frame data structures for your UI face cards
            return {"status": "success", "characters": []} # Handled via faceapp scan if needed

        # Prepare processors chain: Swapper -> Expression Restorer (LivePortrait) -> Enhancer
        processors = ["face_swapper", "expression_restorer"]
        if enhancer != "none":
            processors.append("face_enhancer")

        # Build FaceFusion headless CLI execution command
        cmd = [
            "python", "facefusion.py", "run",
            "--target", target_path,
            --output", output_path,
            "--processors", *processors,
            "--face-swapper-model", swapper_model,
            "--face-enhancer-model", enhancer if enhancer != "none" else "none",
            "--face-enhancer-blend", str(enhancer_blend),
            "--expression-restorer-model", "live_portrait",
            "--expression-restorer-factor", str(expression_factor),
            "--execution-providers", "cuda"
        ]

        print(f"Executing pipeline: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app/facefusion")
        
        if result.returncode != 0:
            return {"error": f"FaceFusion execution failed: {result.stderr}"}

        with open(output_path, "rb") as f:
            output_bytes = f.read()

        return {
            "status": "success",
            "mediatype": "video" if is_video else "image",
            "base64": base64.b64encode(output_bytes).decode("utf-8")
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in [target_path, output_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
