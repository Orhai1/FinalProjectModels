import cv2
import os

import numpy as np


def extract_every_n_seconds(video_path, base_interval=3, duration_threshold=60, scale=0.8):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{video_id}_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Adjust interval based on duration
    if duration > duration_threshold:
        interval_seconds = base_interval * (duration / duration_threshold) ** scale
    else:
        interval_seconds = base_interval

    frame_interval = int(fps * interval_seconds)

    print(f"{video_id}: duration={duration:.1f}s, interval={interval_seconds:.2f}s")

    frame_num = 0
    saved_frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{video_id}_frame_{saved_frame_num:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_num += 1

        frame_num += 1

    cap.release()
    return output_dir

def mse(img1, img2):
    # Mean Squared Error between two images
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return err

def remove_identical_frames(folder_path, mse_threshold=1.0):
    """
    Remove identical frames from a folder of images based on MSE threshold.
    :param folder_path: Path to the folder containing images
    :param mse_threshold: MSE threshold for identifying identical frames
    """
    frame_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not frame_paths:
        print("No frames found.")
        return

    prev_img = cv2.imread(frame_paths[0])
    kept = [frame_paths[0]]
    deleted = 0

    for path in frame_paths[1:]:
        curr_img = cv2.imread(path)

        # Resize to same size if needed
        if curr_img.shape != prev_img.shape:
            curr_img = cv2.resize(curr_img, (prev_img.shape[1], prev_img.shape[0]))

        error = mse(prev_img, curr_img)

        if error < mse_threshold:
            os.remove(path)
            deleted += 1
        else:
            kept.append(path)
            prev_img = curr_img

    # print(f"Done. Removed {deleted} identical frames. Kept {len(kept)}.")


video_path = "/Users/noga/Documents/BarIlan/Year3/Project/LabelingApp/ServerSide/db/downloads/7287950743196257544.mp4"
output_path = extract_every_n_seconds(video_path, 2)
remove_identical_frames(output_path,5)