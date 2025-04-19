import os
import torch
import time
from ultralytics import YOLO

# Paths
BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'

DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')

RUNS_DIR = os.path.join(BASE_DIR, 'runs\\')
TUNE_RUNS = os.path.join(RUNS_DIR, 'tuning\\')

# Load best models from each run
#model_path = os.path.join(TUNE_RUNS, 'iter10\\', 'tune\\', 'weights\\', 'best.pt')
#BEST_HYP = os.path.join(TUNE_RUNS, 'iter10\\', 'tune\\', 'best_hyperparameters.yaml')

#model_path = os.path.join(TUNE_RUNS, 'iter25\\', 'tune\\', 'weights\\', 'best.pt')
#BEST_HYP = os.path.join(TUNE_RUNS, 'iter25\\', 'tune\\', 'best_hyperparameters.yaml')

#model_path = os.path.join(TUNE_RUNS, 'iter50\\', 'tune\\', 'weights\\', 'best.pt')
#BEST_HYP = os.path.join(TUNE_RUNS, 'iter50\\', 'tune\\', 'best_hyperparameters.yaml')

#model_path = os.path.join(TUNE_RUNS, 'iter100\\', 'tune\\', 'weights\\', 'best.pt')
#BEST_HYP = os.path.join(TUNE_RUNS, 'iter100\\', 'tune\\', 'best_hyperparameters.yaml')

def main():    
    # Load model for baseline
    model = YOLO("yolo11m.pt")

    # Limit system memory usage
    torch.set_num_threads(6)

    # Start timer
    t0 = time.time()

    # Run training
    model.train(
        device="cuda:0",
        data=DATA_YAML,
        #cfg=BEST_HYP,
        project="runs/baseline", # baseline, iter10, iter25, iter50, iter100
        verbose=True,
        val=True,
        plots=True,
        save=True,
        imgsz=640,
        epochs=100,
        patience=30
    )

    elapsed = time.time() - t0
    print(f"Finished training on GPU in {elapsed:.2f} seconds")

    # baseline 5239.89 (5239.887646198273) seconds
    # iter10 4640.87 (4640.867440223694) seconds 
    # iter25 4508.46 (4508.464192152023) seconds 
    # iter50 4758.54 (4758.537364959717) seconds
    # iter100 4645.53 (4645.534210920334) seconds

if __name__ == "__main__":
    main()