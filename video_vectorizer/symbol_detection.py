from ultralytics import YOLO
import os

# Load the YOLO model and class names
model = YOLO("../Symbols/symbols_detection_models/yolo11s_v3_updated1/model.pt")
symbol_classes = [model.names[i] for i in range(len(model.names))]


def detect_symbols_in_frames(frames_dir, confidence_threshold=0.5):
    detected = {f'symbol_{cls}': 0 for cls in symbol_classes}

    for filename in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, filename)
        results = model(frame_path)[0]
        for box in results.boxes:
            cls_idx = int(box.cls)
            score = float(box.conf)
            if score > confidence_threshold:
                cls_name = model.names[cls_idx]
                detected[f'symbol_{cls_name}'] = 1

    return detected
