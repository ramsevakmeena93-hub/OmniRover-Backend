"""
Quick local training — fire + smoke detection
Run this in terminal: python quick_train.py

You need your Roboflow API key.
Get it: app.roboflow.com → top right avatar → Settings → API Keys → Copy
"""

import sys, subprocess

# Step 1: Get API key
API_KEY = input("Paste your Roboflow API key and press Enter: ").strip()
if not API_KEY:
    print("No key entered. Exiting.")
    sys.exit(1)

# Step 2: Download dataset
print("\n[1/3] Downloading fire+smoke dataset from Roboflow...")
try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)

    # Your forked project
    workspace = input("Enter your Roboflow workspace name (shown in URL after app.roboflow.com/): ").strip()
    # From your screenshot URL: jys-workspace-czqyt
    if not workspace:
        workspace = "jys-workspace-czqyt"

    project = rf.workspace(workspace).project("smoke-fire-s2oxt-ueoz2")
    dataset = project.version(1).download("yolov8")
    print(f"[1/3] Dataset downloaded to: {dataset.location}")
    data_yaml = f"{dataset.location}/data.yaml"

except Exception as e:
    print(f"Download failed: {e}")
    print("\nTrying with direct dataset path...")
    data_yaml = input("Enter path to data.yaml if you have it downloaded: ").strip()
    if not data_yaml:
        sys.exit(1)

# Step 3: Train
print("\n[2/3] Training YOLOv8s on fire+smoke data (20-30 min on CPU)...")
print("      Press Ctrl+C to stop early — best.pt is saved automatically\n")

from ultralytics import YOLO
import shutil, os

model = YOLO("yolov8s.pt")
results = model.train(
    data=data_yaml,
    epochs=25,
    imgsz=416,
    batch=4,
    name="fire_smoke_local",
    patience=5,
    device="cpu",
    workers=0,
    verbose=True,
)

# Step 4: Copy model
best = "runs/detect/fire_smoke_local/weights/best.pt"
if os.path.exists(best):
    shutil.copy(best, "fire_debris.pt")
    print("\n" + "="*50)
    print("SUCCESS! fire_debris.pt is ready.")
    print("Restart YOLO service — it will use the new model.")
    print("="*50)
else:
    print("Training may still be running. Check runs/detect/fire_smoke_local/weights/")
