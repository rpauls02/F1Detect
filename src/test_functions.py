import os
import torch
import cv2
import time
from ultralytics import YOLO

def run_inference_on_image(model, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    results = model(image_path)
    for result in results:
        annotated = result.plot()
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, annotated)
        print(f"Saved image result in {save_path}")

def run_inference_on_video(model, video_source, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Unable to open video {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.basename(video_source)
    save_path = os.path.join(save_dir, filename)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=(640, 384), verbose=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Saved video result in {save_path}")


def main():
    BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
    RUNS_DIR = os.path.join(BASE_DIR, 'runs\\')
    TRAIN_RUNS = os.path.join(RUNS_DIR, 'training\\')

    DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\', 'test\\')
    TEST_IMG_DIR = os.path.join(DATASET_DIR, 'image\\')
    TEST_CLIPS_DIR = os.path.join(DATASET_DIR, 'video\\')

    RESULTS_DIR = os.path.join(BASE_DIR, 'results\\')
    IMG_RESULTS_DIR = os.path.join(RESULTS_DIR, 'image\\')
    CLIPS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'video\\')

    model_path = os.path.join(TRAIN_RUNS, 'baseline', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter10', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter25', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter50', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter100', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)

    print("\nRunning inference on extracted images...\n")
    for filename in os.listdir(TEST_IMG_DIR):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(TEST_IMG_DIR, filename)
            print(f"Image: {filename}")
            run_inference_on_image(model, image_path, IMG_RESULTS_DIR)

    print("\nRunning inference on extracted video clips...\n")
    for filename in os.listdir(TEST_CLIPS_DIR):
        if filename.lower().endswith(('.mp4')):
            video_path = os.path.join(TEST_CLIPS_DIR, filename)
            print(f"Video: {filename}")
            run_inference_on_video(model, video_path, CLIPS_RESULTS_DIR)

if __name__ == "__main__":
    main()
