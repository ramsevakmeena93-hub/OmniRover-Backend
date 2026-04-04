"""
OmniRover Detection Service
- Model 1: yolov8n.pt -> humans, debris
- Fire/Smoke: HSV color detection (no model needed, instant)
"""
import asyncio, base64, io, json, os
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CONF = 0.20

print("[OmniRover] Loading human detection model...")
model_primary = YOLO("yolov8n.pt")
print("[OmniRover] Ready")

# Load fire model if available
model_fire = None
for p in ["fire_debris.pt", "fire_model.pt"]:
    if os.path.exists(p):
        model_fire = YOLO(p)
        print(f"[OmniRover] Fire model loaded: {p} | classes: {model_fire.names}")
        break

PRIMARY_MAP = {
    "person": "human",
    "rock": "debris", "rubble": "debris", "chair": "debris",
    "couch": "debris", "bed": "debris", "suitcase": "debris",
    "bottle": "gas", "vase": "gas",
}

FIRE_MAP = {
    "fire": "fire", "Fire": "fire", "smoke": "fire",
    "Smoke": "fire", "flame": "fire", "2.": "fire",
}

COLORS = {
    "human": "#00ff88",
    "fire":  "#ff2d2d",
    "gas":   "#ffaa00",
    "debris":"#3b82f6",
}


def run_primary(img_np):
    """Detect humans and debris using YOLOv8n"""
    results = model_primary(img_np, verbose=False, conf=CONF)
    dets, flags = [], {"human": False, "fire": False, "gas": False, "debris": False}
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            name = model_primary.names[int(box.cls[0])].lower()
            cat = PRIMARY_MAP.get(name)
            if not cat:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            flags[cat] = True
            dets.append({
                "class": cat, "label": name, "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "color": COLORS[cat], "source": "yolo"
            })
    return dets, flags


def run_fire_model(img_np):
    """Use trained fire model if available"""
    if not model_fire:
        return [], {"human": False, "fire": False, "gas": False, "debris": False}
    results = model_fire(img_np, verbose=False, conf=CONF)
    dets, flags = [], {"human": False, "fire": False, "gas": False, "debris": False}
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            name = model_fire.names[int(box.cls[0])]
            cat = FIRE_MAP.get(name) or FIRE_MAP.get(name.lower())
            if not cat:
                if any(k in name.lower() for k in ["fire", "smoke", "flame"]):
                    cat = "fire"
                else:
                    continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            flags[cat] = True
            dets.append({
                "class": cat, "label": name, "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "color": COLORS.get(cat, "#ff2d2d"), "source": "fire_model"
            })
    return dets, flags


def detect_fire_color(img_np):
    """
    HSV color-based fire & smoke detection.
    Fire = bright orange/red/yellow pixels in clusters.
    Smoke = large gray/white diffuse regions.
    No model needed — works instantly.
    """
    dets = []
    flags = {"human": False, "fire": False, "gas": False, "debris": False}
    h_img, w_img = img_np.shape[:2]
    total_pixels = h_img * w_img

    try:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # ── FIRE: only very bright, saturated orange-red pixels ──────
        # Tight range — must be highly saturated AND very bright (actual flame)
        fire_mask1 = cv2.inRange(hsv, np.array([0,  200, 200]), np.array([18, 255, 255]))
        fire_mask2 = cv2.inRange(hsv, np.array([165,200, 200]), np.array([180,255, 255]))
        fire_mask  = cv2.bitwise_or(fire_mask1, fire_mask2)

        # Clean up noise
        k = np.ones((7, 7), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, k)
        fire_mask = cv2.dilate(fire_mask, k, iterations=2)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_fire_area = total_pixels * 0.012  # 1.2% of frame minimum

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_fire_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            roi = img_np[y:y+bh, x:x+bw]
            if roi.size == 0:
                continue
            # Verify: fire must be very red AND bright AND more red than green
            mean_r = float(np.mean(roi[:, :, 0]))
            mean_g = float(np.mean(roi[:, :, 1]))
            mean_b = float(np.mean(roi[:, :, 2]))
            # Strict checks: very bright red, red >> green, red >> blue
            if mean_r < 180:
                continue  # not bright enough
            if mean_r < mean_g + 60:
                continue  # not red/orange enough (skin tone has r~g)
            if mean_b > 100:
                continue  # too much blue (not fire)
            conf = min(0.93, (area / total_pixels) * 18)
            flags["fire"] = True
            dets.append({
                "class": "fire", "label": "fire",
                "confidence": round(conf, 3),
                "bbox": [x, y, x + bw, y + bh],
                "color": "#ff2d2d",
                "source": "color"
            })

        # Smoke detection disabled — too many false positives indoors
        # Enable only when rover is in outdoor/disaster environment
        # s_contours = []

    except Exception as e:
        print(f"[COLOR] Error: {e}")

    return dets, flags


def process_frame(b64: str):
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    det1, f1 = run_primary(img_np)
    det2, f2 = run_fire_model(img_np)
    det3, f3 = detect_fire_color(img_np)

    all_det = det1 + det2 + det3
    human  = f1["human"]  or f2["human"]  or f3["human"]
    fire   = f1["fire"]   or f2["fire"]   or f3["fire"]
    gas    = f1["gas"]    or f2["gas"]    or f3["gas"]
    debris = f1["debris"] or f2["debris"] or f3["debris"]
    risk   = "CRITICAL" if (fire or gas) else "WARNING" if (human or debris) else "SAFE"

    return {
        "detections":    all_det,
        "humanDetected": human,
        "fireDetected":  fire,
        "gasDetected":   gas,
        "debrisDetected":debris,
        "risk":          risk,
        "count":         len(all_det),
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "frame" and msg.get("image"):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_frame, msg["image"]
                )
                await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "human_model": "yolov8n.pt",
        "fire_model": model_fire is not None,
        "color_detection": True,
    }


if __name__ == "__main__":
    import uvicorn
    print("[OmniRover] ws://localhost:5001/ws")
    print("[OmniRover] Fire detection: HSV color + YOLO (if model present)")
    uvicorn.run(app, host="0.0.0.0", port=5001)
