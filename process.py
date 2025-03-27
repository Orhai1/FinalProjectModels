# process_all_videos.py

import os
from extract_frames import extract_frames

def process_videos_in_folder(folder_path, diff_threshold):
    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [f for f in os.listdir(folder_path)
              if os.path.splitext(f)[1].lower() in video_extensions]

    if not videos:
        print("No video files found in folder.")
        return

    for video in videos:
        video_path = os.path.join(folder_path, video)
        folder_name = extract_frames(video_path, diff_threshold=diff_threshold)
        print(f"Created: {folder_name}\n")

if __name__ == "__main__":
    folder_path = "videos"
    process_videos_in_folder(folder_path, diff_threshold=30)
