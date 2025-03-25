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
HYP_YAML = os.path.join(SRC_DIR, 'base_hyp.yaml')

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
results = model.tune(
    device='cuda',
    data=DATA_YAML,
    space=hyp_space,
    project="runs/iter30",
    name="exp",
    plots=True,
    save=True,
    batch=16,
    imgsz=640,
    epochs=10,
    iterations=30,
    verbose=True,
    val=True,
    single_cls=True
)

print('Finished tuning on GPU in ', time.time() - t0, ' seconds')