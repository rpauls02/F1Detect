import os
import cv2
import random

BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
print(BASE_DIR)
TEST_VIDEOS = os.path.join(BASE_DIR, 'raw\\', 'broadcast\\', 'test\\')
print(TEST_VIDEOS)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
print(DATASET_DIR)
TEST_IMGS_DIR = os.path.join(DATASET_DIR, 'test\\', 'images\\')
print(TEST_IMGS_DIR)

def extract_frames(video_file, output_folder, num_frames=30):
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
    for video_file in os.listdir(TEST_VIDEOS):
        video_path = os.path.join(TEST_VIDEOS, video_file)
        extract_frames(video_path, TEST_IMGS_DIR)

if __name__ == "__main__":
    main()