"""
Train YOLOv8 Fire + Smoke + Debris Detection Model
====================================================
Step 1: Get FREE Roboflow API key at https://roboflow.com (takes 30 seconds)
Step 2: Paste your key below
Step 3: Run: python train_fire_model.py
Step 4: Training takes ~20-40 min on CPU

The trained model will be saved as fire_debris.pt
"""

from roboflow import Roboflow
from ultralytics import YOLO
import os, shutil

# ── PASTE YOUR FREE ROBOFLOW API KEY HERE ──────────────────────
API_KEY = "YOUR_API_KEY_HERE"
# Get it free at: https://app.roboflow.com → Settings → API Keys
# ───────────────────────────────────────────────────────────────

if API_KEY == "YOUR_API_KEY_HERE":
    print("=" * 60)
    print("ERROR: Please add your Roboflow API key!")
    print("1. Go to https://app.roboflow.com")
    print("2. Sign up free")
    print("3. Settings → API Keys → Copy key")
    print("4. Paste it in this file where it says YOUR_API_KEY_HERE")
    print("=" * 60)
    exit(1)

print("[TRAIN] Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)

# Download fire+smoke dataset (public, free)
print("[TRAIN] Downloading fire+smoke dataset...")
project = rf.workspace("mohamedmustafa").project("fire-and-smoke-detection-jaster")
dataset = project.version(2).download("yolov8")
print(f"[TRAIN] Dataset: {dataset.location}")

# Also download debris/rubble dataset
print("[TRAIN] Downloading debris dataset...")
try:
    project2 = rf.workspace("disaster-response").project("rubble-detection")
    dataset2 = project2.version(1).download("yolov8")
    print(f"[TRAIN] Debris dataset: {dataset2.location}")
except:
    print("[TRAIN] Using fire dataset only")
    dataset2 = None

# Train on YOLOv8s (better than nano for custom classes)
print("[TRAIN] Starting training (this takes 20-40 min on CPU)...")
model = YOLO("yolov8s.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=30,
    imgsz=416,
    batch=4,
    name="fire_debris_v1",
    patience=5,
    device="cpu",
    workers=0,
    verbose=False,
)

# Copy best model
best = "runs/detect/fire_debris_v1/weights/best.pt"
if os.path.exists(best):
    shutil.copy(best, "fire_debris.pt")
    print("\n" + "="*60)
    print("SUCCESS! Model saved as: server/yolo/fire_debris.pt")
    print("Restart YOLO service — it will auto-detect the new model")
    print("="*60)
else:
    print("Training failed — check errors above")
