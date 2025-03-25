import os
import yaml
import torch
import sys
import time
from ultralytics import YOLO

# Paths
BASE_DIR = 'P:/GitHub/F1-Car-Detection/'

DATASET_DIR = os.path.join(BASE_DIR, 'dataset/')
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')

SRC_DIR = os.path.join(BASE_DIR, 'src/')

# Limit system memory usage
torch.set_num_threads(6)

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = YOLO("yolo11m.pt")
if not model:
    print(f"{model} not found. Exiting.")
    sys.exit(1)

t0 = time.time()

# Run tuning
results = model.train(
    device='cuda',
    data=DATA_YAML,
    project="runs/final",
    name="exp",
    plots=True,
    save=True,
    batch=16,
    imgsz=640,
    epochs=100,
    verbose=True,
    val=True
)

print('Finished tuning on GPU in ', time.time() - t0, ' seconds')