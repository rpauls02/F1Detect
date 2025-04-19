F1-Car-Detection/
├── .venv/
├── dataset/
│   ├── test/
|   |   ├── real/
│   │   |   ├── image/
│   │   |   └── video/
|   |   ├── synthetic/
│   │   |   ├── image/
│   │   |   └── video/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── raw/
|   ├── synthetic/
|   |   ├── test/
├── src/        
│   ├── extract_clips.py
│   ├── extract_frames.py
│   ├── hyp.yaml
│   ├── test_functions.py
│   ├── yolo11m_train.py
│   └── yolo11m_tune.py
