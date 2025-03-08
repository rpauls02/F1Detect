import os
import cv2
import random

def load_raw_video(video_folder):
    if not os.path.exists(video_folder):
        print(f"Directory '{video_folder}' does not exist")
        return {}

    video_files = [file for file in os.listdir(video_folder) if file.endswith('.mp4')]
    
    if not (video_files):
        print(f"No data found in '{video_folder}'")
        return {}
    
    video_data = {video: os.path.join(video_folder, video) for video in video_files}
    print(f"Loaded {len(video_data)} videos from '{video_folder}'.")
    return video_data

def extract_clips(video_file, output_folder, duration=10, num_clips=10):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if f_total <= 0 or fps <= 0:
        print(f"Error reading video: {video_file}")
        cap.release()
        return

    fpc = int(fps * duration)  # Frames per clip
    video_name = os.path.basename(video_file).split('.')[0]

    # Define valid range for random clip selection (ignoring first & last 10 seconds)
    start_frame_min = int(fps * 10)
    start_frame_max = int(f_total - fps * 10 - fpc)

    if start_frame_max <= start_frame_min:
        print(f"Video {video_file} is too short.")
        cap.release()
        return

    # Select 10 random start frames within the valid range
    start_frames = sorted(random.sample(range(start_frame_min, start_frame_max), min(num_clips, (start_frame_max - start_frame_min) // fpc)))

    print(f"Extracting {len(start_frames)} clips from {video_file}")

    for c_index, start_frame in enumerate(start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to the random start frame

        output_clip_path = os.path.join(output_folder, f"{video_name}_clip_{c_index:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_clip_path, fourcc, int(fps), (f_width, f_height))

        print(f"Creating clip {c_index}: Start frame {start_frame}, saving to {output_clip_path}")

        for _ in range(fpc):
            ret, frame = cap.read()
            if not ret:
                print(f"Unexpected end of video {video_file} at frame {start_frame + _}")
                break
            out.write(frame)

        out.release()
        print(f"Saved: {output_clip_path}")

    cap.release()
    print(f"Successfully extracted {len(start_frames)} clips from {video_file} \n")