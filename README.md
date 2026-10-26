# F1 Car Detection

Real-time object detection system for identifying Formula 1 race cars in video footage, trained on a Roboflow dataset combining video-game and real-world clips.

## Overview

Built an end-to-end detection pipeline using **YOLOv11** (Ultralytics), covering dataset construction, CUDA-accelerated training, and evolutionary hyperparameter search. The final model runs at real-time speeds on GPU, benchmarked across a held-out set of 1,100+ frames.

## Results

| Metric | Value |
|---|---|
| Inference speed | ~60 FPS (real-time) |
| Benchmark set | 1,100+ frames |

## Approach

- **Dataset**: single-class (`race car`) detection dataset built from a mix of video-game and real-world F1 footage, hosted and versioned via Roboflow ([source](https://universe.roboflow.com/f1-vehicle-detection/f1-car-dataset-kkvsm/7)).
- **Preprocessing**: `src/extract_frames.py` and `src/extract_clips.py` extract and prepare frame/clip data from raw video sources.
- **Model**: YOLOv11m, trained with CUDA acceleration (`src/yolo11m_train.py`).
- **Hyperparameter tuning**: evolutionary search across multiple iteration budgets (10/50 experiments) via `src/yolo11m_tune.py`, with results compared against baseline training.
- **Evaluation**: `src/test_functions.py` benchmarks the trained model against real and synthetic test data.

## Project structure

```
F1Detect/
├── dataset/            # Roboflow-sourced dataset (train/val/test splits, YOLO format)
├── src/
│   ├── extract_frames.py
│   ├── extract_clips.py
│   ├── yolo11m_train.py
│   ├── yolo11m_tune.py
│   ├── hyp.yaml
│   └── test_functions.py
```

## Running it

See `instructions.txt` for full step-by-step commands. In short:

1. Set `BASE_DIR` to your local project path in the relevant script.
2. Run frame/clip extraction scripts to (re)generate synthetic test data.
3. Run `yolo11m_train.py` for baseline training, or `yolo11m_tune.py` for hyperparameter search.
4. Run `test_functions.py` to evaluate against real and synthetic test sets.

## Tech stack

Python, PyTorch, Ultralytics YOLOv11, OpenCV, CUDA, Roboflow.
