import os
import torch
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

# Dataset Paths
DATASET_PATH = "./dataset/"
TEST_PATH = os.path.join(DATASET_PATH, "test/images")

# Model Paths
MODEL_WEIGHTS = "./runs/train/exp/weights/best.pt"

def test_image(img_path, result_path, model):
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} does not exist.")
        return
    
    result = model(img_path)
    result.print()
    img = np.squeeze(result.render())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_name = os.path.basename(img_path)
    pcd_img_name = img_name + '_processed.jpg'
    save_path = os.path.join(result_path, pcd_img_name)
    cv2.imwrite(save_path, img)
    print(f"{img_name} processed. Saved at {save_path}")

        
def test_video(video_path, result_path, model):
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} does not exist.")
        return
    
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    pcd_vid_name = video_name + '_processed.jpg'
    save_path = os.path.join(result_path, pcd_vid_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_width, target_height = 640, 360
    upscale_width, upscale_height = 1920, 1080
    writer = cv2.VideoWriter(save_path, fourcc, 60, (upscale_width, upscale_height))

    
    while cap.isOpened():
        isframe, frame = cap.read()
        if isframe:
            # Process at 640p resolution
            frame_resized = cv2.resize(frame, (target_width, target_height))
            result = model(frame_resized)
            interim = np.squeeze(result.render())
            
            # Upscale back to 1080p before saving
            interim_upscaled = cv2.resize(interim, (upscale_width, upscale_height), interpolation=cv2.INTER_CUBIC)

            try:
                writer.write(interim_upscaled)
            except:
                print('Could not write to file')
            if cv2.waitKey(10) & 0xFF == 27:
                break
        else:
            print(f"{video_name} processed")
            break 
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"{video_name} processed. Saved at {save_path}")
    
