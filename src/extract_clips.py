import os
import cv2
import random

'''
Since GUI elements of the F1 23 game can be completely removed in the replay feature,
this custom function could be used to extract clips from the recording.

Unlike the game, real world footage often contains a lot of graphics that cannot be removed 
unless raw footage can be retrieved (nearly impossible seemingly). Due to this, manual 
collection of frames and clips is much more convenient, and can be cropped to remove graphics 
interfering with racing footage useful for testing model
'''

BASE_DIR = 'P:\\GitHub\\F1-Car-Detection\\'
TEST_VID_DIR = os.path.join(BASE_DIR, 'raw\\', 'broadcast\\', 'test\\')

DATASET_DIR = os.path.join(BASE_DIR, 'dataset\\')
TEST_CLIPS_DIR = os.path.join(DATASET_DIR, 'test\\', 'video\\')

def extract_clips(video_file, output_folder, duration=5, num_clips=5):
    """Extract random video clips of given duration from a video file."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[Error] Failed to open video: {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = int(fps * duration)

    # Validate video properties
    if fps <= 0 or total_frames <= 0:
        print(f"[Error] Invalid video or unreadable properties: {video_file}")
        cap.release()
        return

    # Ensure enough frames (skip first/last 10s)
    buffer_frames = int(fps * 10)
    valid_range = total_frames - 2 * buffer_frames - frames_per_clip
    if valid_range <= 0:
        print(f"[Error] Video too short to extract clips: {video_file}")
        cap.release()
        return

    possible_starts = range(buffer_frames, total_frames - buffer_frames - frames_per_clip)
    start_frames = sorted(random.sample(possible_starts, min(num_clips, len(possible_starts) // frames_per_clip)))

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    print(f"[Info] Extracting {len(start_frames)} clips from: {video_name}")

    for idx, start_frame in enumerate(start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_path = os.path.join(output_folder, f"{video_name}_clip_{idx:03d}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"[Clip {idx+1}] Starting at frame {start_frame} → {output_path}")

        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                print(f"[Warning] Early termination at frame {start_frame + _} for clip {idx}")
                break
            out.write(frame)

        out.release()

    cap.release()
    print(f"[Done] All clips extracted to: {output_folder}")

def main():
    for video_file in os.listdir(TEST_VID_DIR):
        video_path = os.path.join(TEST_VID_DIR, video_file)
        if os.path.isfile(video_path):
            extract_clips(video_path, TEST_CLIPS_DIR)

if __name__ == "__main__":
    main()