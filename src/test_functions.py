import os
import torch
import cv2
import time
from ultralytics import YOLO

def run_inference_on_image(model, image_path):
    results = model(image_path)
    for result in results:
        result.show()

def run_inference_on_video(model, video_source, window_width=1280, window_height=720):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return
        
    # Create named window with resize property
    cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Inference', window_width, window_height)
    
    while True:
        start_time = time.time()  # Move time measurement before frame reading
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model.predict(source=frame, imgsz=(640, 384), verbose=False)
        
        # Visualize the result
        annotated_frame = results[0].plot()
        
        inference_time = time.time() - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        
        # Overlay controls and FPS
        cv2.putText(annotated_frame, "ESC: Quit | SPACE: Pause/Play", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
       
        cv2.imshow('YOLO Inference', annotated_frame)
        
        # Press 'esc' to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
    RUNS_DIR = os.path.join(BASE_DIR, 'runs\\')
    TRAIN_RUNS = os.path.join(RUNS_DIR, 'training\\')

    torch.set_num_threads(6)

    model_path = os.path.join(TRAIN_RUNS, 'baseline', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter10', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter25', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter50', 'train', 'weights', 'best.pt')
    #model_path = os.path.join(TRAIN_RUNS, 'iter100', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)

    # source = "P:\\GitHub\\F1-Car-Detection\\raw\\images\\f1_car.jpg"
    source = "P:\\GitHub\\F1-Car-Detection\\raw\\broadcast\\albertpark_midday_clear.mp4"

    if isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
        run_inference_on_image(model, source)
    else:
        run_inference_on_video(model, source)

if __name__ == "__main__":
    main()
