"""
Train YOLOv8 on Fire + Smoke + Debris dataset
Uses Roboflow public dataset (no API key needed for public datasets)

Run: python train_model.py
Time: ~30-60 min on CPU, ~5-10 min on GPU
"""

import os, subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--user", "-q"])

# Install roboflow
try:
    import roboflow
except ImportError:
    print("Installing roboflow...")
    install("roboflow")
    import roboflow

from roboflow import Roboflow
from ultralytics import YOLO

# ── Download public fire+smoke dataset from Roboflow ──────────
print("[TRAIN] Downloading fire+smoke+debris dataset...")

rf = Roboflow(api_key="YOUR_API_KEY")  # Free at roboflow.com

# Public dataset: Fire and Smoke Detection
project = rf.workspace("school-tvtyj").project("fire-smoke-detection-2")
dataset = project.version(1).download("yolov8")

print(f"[TRAIN] Dataset downloaded to: {dataset.location}")

# ── Train ─────────────────────────────────────────────────────
print("[TRAIN] Starting training...")
model = YOLO("yolov8s.pt")  # Start from pretrained weights

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="fire_debris_model",
    patience=10,
    device="cpu",  # Change to 0 if you have GPU
)

print("[TRAIN] Done! Model saved to runs/detect/fire_debris_model/weights/best.pt")
print("[TRAIN] Copy it: copy runs/detect/fire_debris_model/weights/best.pt fire_debris.pt")
