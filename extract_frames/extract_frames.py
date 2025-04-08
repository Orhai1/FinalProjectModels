# frame_extractor.py

import cv2
import os
import numpy as np

def extract_frames(video_path, diff_threshold=50):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = f"{video_id}_frames"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read {video_path}")
        return None

    last_saved_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_saved_gray = cv2.GaussianBlur(last_saved_gray, (21, 21), 0)

    saved_count = 0
    frame_index = 0

    cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count:04d}.jpg"), frame)
    saved_count += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(last_saved_gray, gray)
        if (np.count_nonzero(diff) / diff.size) * 100 >= diff_threshold:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count:04d}.jpg"), frame)
            saved_count += 1
            last_saved_gray = gray

    cap.release()
    # print(f"✅ {video_path} → {saved_count} frames → {output_folder}")
    return output_folder
