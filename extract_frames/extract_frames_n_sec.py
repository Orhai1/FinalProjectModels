import cv2
import os

def extract_every_n_seconds(video_path, interval_seconds=3):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{video_id}_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_num = 0
    saved_frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_frame_num:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_num += 1

        frame_num += 1

    cap.release()
    return output_dir
