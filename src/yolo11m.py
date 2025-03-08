import os
import yaml
import torch
import shutil
import sys
from ultralytics import YOLO

BASE_DIR = 'P:/GitHub/F1-Car-Detection/'

DATASET_DIR = os.path.join(BASE_DIR, 'dataset/')
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')

SRC_DIR = os.path.join(BASE_DIR, 'src/')
HYP_YAML = os.path.join(SRC_DIR, 'hyp.yaml')

# Load hyperparameter space
with open(HYP_YAML, "r") as f:
    hyp_space = yaml.safe_load(f)

# Check for string values
for key, value in hyp_space.items():
    if isinstance(value, list):
        hyp_space[key] = [float(v) for v in value]  # Convert to floats
    elif isinstance(value, str):  
        try:
            hyp_space[key] = float(value)  # Convert standalone strings
        except ValueError:
            pass  # Ignore if conversion fails

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = YOLO("yolo11m.pt").to(device)
if not model:
    print("Model not found. Exiting.")
    sys.exit(1)
    
print(f"yolo11m loaded on {device}")

results = model.tune(
    device=device,
    data=DATA_YAML,
    space=hyp_space,
    project="runs",
    name="exp",
    plots=True,
    save=True,
    batch=32,
    imgsz=640,
    optimizer="SGD",
    patience=50,
    epochs=10,
    iterations=10
)