import os
import cv2
import random

BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
print(BASE_DIR)
TEST_VIDEOS = os.path.join(BASE_DIR, 'raw\\', 'broadcast\\', 'test\\')
print(TEST_VIDEOS)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
print(DATASET_DIR)
TEST_CLIPS_DIR = os.path.join(DATASET_DIR, 'test\\', 'clips\\')
print(TEST_CLIPS_DIR)

def extract_clips(video_file, output_folder, duration=5, num_clips=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = int(fps * duration)

    # Validate video and duration
    if fps <= 0 or total_frames <= 0:
        print(f"Invalid video or unreadable properties: {video_file}")
        cap.release()
        return

    # Ensure we have enough room to extract clips (skip first/last 10s)
    buffer_frames = int(fps * 10)
    valid_range = total_frames - 2 * buffer_frames - frames_per_clip
    if valid_range <= 0:
        print(f"Video too short to extract clips: {video_file}")
        cap.release()
        return

    # Generate unique random start frames
    possible_starts = range(buffer_frames, total_frames - buffer_frames - frames_per_clip)
    start_frames = sorted(random.sample(possible_starts, min(num_clips, len(possible_starts) // frames_per_clip)))

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    print(f"Extracting {len(start_frames)} clips from: {video_file}")

    for idx, start in enumerate(start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        output_path = os.path.join(output_folder, f"{video_name}_clip_{idx:03d}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"Clip {idx+1}: Starting at frame {start} → {output_path}")

        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                print(f"Early termination at frame {start + _} for clip {idx}")
                break
            out.write(frame)

        out.release()

    cap.release()
    print(f"All clips extracted and saved to: {output_folder}")

def main():
    for video_file in os.listdir(TEST_VIDEOS):
        video_path = os.path.join(TEST_VIDEOS, video_file)
        extract_clips(video_path, TEST_CLIPS_DIR)

if __name__ == "__main__":
    main()