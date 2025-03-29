import os
import yaml
import torch
from ultralytics import YOLO

def main():
    # Paths
    BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'

    DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
    DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')

    SRC_DIR = os.path.join(BASE_DIR, 'src\\')
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

    # Load model
    model = YOLO("yolo11m.pt")

    # Run tuning
    model.tune(
        device="cuda:0",
        data=DATA_YAML,
        space=hyp_space,
        project="runs/tuning/iter100",
        name="tune",
        verbose=True,
        val=True,
        single_cls=True,
        plots=True,
        save=True,
        batch=16,
        imgsz=640,
        epochs=10,
        iterations=100
    )

    # iter10 5907.35 seconds
    # iter25 13950.71 seconds
    # iter50 28826.63 seconds
    # iter100 epoch10 54890.41 seconds


if __name__ == "__main__":
    main()