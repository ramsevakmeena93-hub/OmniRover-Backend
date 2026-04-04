"""
Download pre-trained fire/smoke/debris detection models
Run: python download_model.py
"""
import urllib.request, os

models = {
    # Fire & Smoke detection model (trained on fire dataset)
    "fire_smoke.pt": "https://github.com/spacewalk01/yolov9-fire-detection/releases/download/v1.0/best.pt",
}

for name, url in models.items():
    if os.path.exists(name):
        print(f"[OK] {name} already exists")
        continue
    print(f"[DOWNLOADING] {name} from {url}")
    try:
        urllib.request.urlretrieve(url, name)
        print(f"[DONE] {name}")
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        print("Try manual download or use Option 2 below")

print("\nTo use: set YOLO_MODEL=fire_smoke.pt in server/yolo/.env")
