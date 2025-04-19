import os
import yaml
import torch
from ultralytics import YOLO

# Paths
BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'

DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
SRC_DIR = os.path.join(BASE_DIR, 'src\\')

DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')
HYP_YAML = os.path.join(SRC_DIR, 'hyp.yaml')

def load_yaml(hyp_yaml_path):
    # Load hyperparameter space from a YAML file
    with open(hyp_yaml_path, "r") as f:
        hyp_space = yaml.safe_load(f)

    # Convert lists and string values to floats
    for key, value in hyp_space.items():
        if isinstance(value, list):
            hyp_space[key] = [float(v) for v in value]
        elif isinstance(value, str):
            try:
                hyp_space[key] = float(value)
            except ValueError:
                pass  # Keep original if conversion fails
    return hyp_space

def main():
    # Load hyperparameter space
    hyp_space = load_yaml(HYP_YAML)

    # Limit system memory usage
    torch.set_num_threads(6)

    # Load model
    model = YOLO("yolo11m.pt")

    # Run tuning
    model.tune(
        device="cuda:0",
        data=DATA_YAML,
        space=hyp_space,
        project="runs/tuning/iter10", # iter10, iter25, iter50, iter100
        name="tune",
        verbose=True,
        val=True,
        single_cls=True,
        plots=True,
        save=True,
        batch=16,
        imgsz=640,
        epochs=10,
        iterations=10 # 10, 25, 50, 100
    )

    # iter10 5907.35 seconds
    # iter25 13950.71 seconds
    # iter50 28826.63 seconds
    # iter100 epoch10 54890.41 seconds

if __name__ == "__main__":
    main()