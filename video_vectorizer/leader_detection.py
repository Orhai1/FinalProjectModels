import os
import joblib
from PIL import Image

leader_model = joblib.load("leader_model.pkl")
leader_classes = ['arafat', 'haniyeh']  # Adjust if dynamic

def extract_features_for_leader_model(image):
    # Define your own image-to-feature logic
    return [...]

def detect_leaders_in_frames(frames_dir):
    detected = {f'leader_{name}': 0 for name in leader_classes}

    for filename in os.listdir(frames_dir):
        image = Image.open(os.path.join(frames_dir, filename)).convert("RGB")
        features = extract_features_for_leader_model(image)
        probs = leader_model.predict_proba([features])[0]
        for i, leader in enumerate(leader_classes):
            if probs[i] > 0.5:
                detected[f'leader_{leader}'] = 1

    return detected
