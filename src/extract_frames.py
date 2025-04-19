import os
import cv2
import random

'''
Since GUI elements of the F1 23 game can be completely removed in the replay feature,
this custom function could be used to extract frames from the recording.

Unlike the game, real world footage often contains a lot of graphics that cannot be removed 
unless raw footage can be retrieved (nearly impossible seemingly). Due to this, manual 
collection of frames and clips is much more convenient, and can be cropped to remove graphics 
interfering with racing footage useful for testing model
'''

BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
TEST_VID_DIR = os.path.join(BASE_DIR, 'raw\\', 'synthetic\\', 'test\\')

DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
TEST_IMG_DIR = os.path.join(DATASET_DIR, 'test\\', 'synthetic\\', 'image\\')

def extract_frames(video_file, output_folder, num_frames=50):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"No frames found in video: {video_file}")
        cap.release()
        return

    # Skip first and last 10 frames as a buffer
    buffer = 10
    if total_frames <= 2 * buffer:
        print(f"Video too short to extract frames: {video_file}")
        cap.release()
        return

    frame_indices = sorted(random.sample(range(buffer, total_frames - buffer), min(num_frames, total_frames - 2 * buffer)))
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    print(f"Extracting {len(frame_indices)} frames from: {video_file}")

    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_no}")
            continue

        frame_path = os.path.join(output_folder, f"{video_name}_frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame {frame_no} → {frame_path}")

    cap.release()
    print(f"Done extracting {len(frame_indices)} frames to: {output_folder}")

def main():
    for video_file in os.listdir(TEST_VID_DIR):
        video_path = os.path.join(TEST_VID_DIR, video_file)
        extract_frames(video_path, TEST_IMG_DIR)

if __name__ == "__main__":
    main()