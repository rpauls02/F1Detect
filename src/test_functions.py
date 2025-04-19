import os
import cv2
import time
from ultralytics import YOLO

# Paths
BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
RUNS_DIR = os.path.join(BASE_DIR, 'runs\\')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\', 'test\\')

TRAIN_RUNS = os.path.join(RUNS_DIR, 'training\\')

SYN_TEST_IMG_DIR = os.path.join(DATASET_DIR, 'synthetic\\', 'image\\')
SYN_TEST_CLIPS_DIR = os.path.join(DATASET_DIR, 'synthetic\\', 'video\\')
SYN_RESULTS_DIR = os.path.join(BASE_DIR, 'results\\', 'synthetic\\', 'iter100\\')
SYN_IMG_RESULTS_DIR = os.path.join(SYN_RESULTS_DIR, 'image\\')
SYN_CLIPS_RESULTS_DIR = os.path.join(SYN_RESULTS_DIR, 'video\\')

REAL_TEST_IMG_DIR = os.path.join(DATASET_DIR, 'real\\', 'image\\', 'cropped\\')
REAL_TEST_CLIPS_DIR = os.path.join(DATASET_DIR, 'real\\', 'video\\')
REAL_RESULTS_DIR = os.path.join(BASE_DIR, 'results\\', 'real\\', 'iter100\\')
REAL_IMG_RESULTS_DIR = os.path.join(REAL_RESULTS_DIR, 'image\\')
REAL_CLIPS_RESULTS_DIR = os.path.join(REAL_RESULTS_DIR, 'video\\')

def run_inference_on_image(model, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    results = model.predict(image, verbose=True, device="cuda")

    for result in results:
        annotated_image = result.plot()
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, annotated_image)

    print("Image inference complete")

def run_inference_on_video(model, video_source, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Unable to open video {video_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_shape = (width, height)

    filename = os.path.basename(video_source)
    filename_no_ext = os.path.splitext(filename)[0]
    save_path = os.path.join(save_dir, filename)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, original_shape)

    # Lists to store per-frame timings
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        t1 = t0

        results = model.predict(source=frame, verbose=False, device="cuda", save_txt=True)
        t2 = time.time()

        result = results[0]
        annotated_frame = result.plot()
        t3 = time.time()

        out.write(annotated_frame)

        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t0) * 1000)

    cap.release()
    out.release()
    print(f"Saved video result in {save_path}")

    # Calculate averages
    avg_pre = sum(preprocess_times) / len(preprocess_times)
    avg_inf = sum(inference_times) / len(inference_times)
    avg_post = sum(postprocess_times) / len(postprocess_times)

    # Save timing metrics into .txt file
    timing_txt_path = os.path.join(save_dir, f"{filename_no_ext}_metrics.txt")
    with open(timing_txt_path, "w") as f:
        f.write(f"Average timings per frame (ms):\n")
        f.write(f"Preprocess: {avg_pre:.2f} ms\n")
        f.write(f"Inference: {avg_inf:.2f} ms\n")
        f.write(f"Postprocess: {avg_post:.2f} ms\n")

    print(f"Saved timings in {timing_txt_path}")

def main():
    # Load best model
    model_path = os.path.join(TRAIN_RUNS, 'iter100\\', 'train\\', 'weights\\', 'best.pt')
    model = YOLO(model_path)

    print("\nRunning inference on real test image data...\n")
    for filename in os.listdir(REAL_TEST_IMG_DIR):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(REAL_TEST_IMG_DIR, filename)
            print(f"Image: {filename}")
            run_inference_on_image(model, image_path, REAL_IMG_RESULTS_DIR)
    
    print("\nRunning inference on synthetic test image data...\n")
    for filename in os.listdir(SYN_TEST_IMG_DIR):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(SYN_TEST_IMG_DIR, filename)
            print(f"Image: {filename}")
            run_inference_on_image(model, image_path, SYN_IMG_RESULTS_DIR)

    print("\nRunning inference on real test video data...\n")
    for filename in os.listdir(REAL_TEST_CLIPS_DIR):
        if filename.lower().endswith(('.mp4')):
            video_path = os.path.join(REAL_TEST_CLIPS_DIR, filename)
            print(f"Video: {filename}")
            run_inference_on_video(model, video_path, REAL_CLIPS_RESULTS_DIR)

    print("\nRunning inference on synthetic test video data...\n")
    for filename in os.listdir(SYN_TEST_CLIPS_DIR):
        if filename.lower().endswith(('.mp4')):
            video_path = os.path.join(SYN_TEST_CLIPS_DIR, filename)
            print(f"Video: {filename}")
            run_inference_on_video(model, video_path, SYN_CLIPS_RESULTS_DIR)

if __name__ == "__main__":
    main()
