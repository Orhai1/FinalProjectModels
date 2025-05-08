import os
from deepface import DeepFace
from joblib import load

# Adjust path to model files relative to this script location
base_path = "../LeadersRecognition/DeepFaceRandomForestModel"
model = load(os.path.join(base_path, "model_randomforest.pkl"))
scaler = load(os.path.join(base_path, "scaler.pkl"))
le = load(os.path.join(base_path, "label_encoder.pkl"))

# Classes you want to detect
leader_classes = ['Arafat', 'AbuGihad', 'AbuMazen', 'Aruri', 'Hania', 'HosainHasheh', 'Sinwar', 'Yassin']
leader_classes_lower = [name.lower() for name in leader_classes]

def extract_faces_from_frame(image_path):
    try:
        faces = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
        return [face["embedding"] for face in faces]
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return []

def detect_leaders_in_frames(frames_dir):
    detected = {f'leader_{name}': 0 for name in leader_classes}

    for filename in os.listdir(frames_dir):
        image_path = os.path.join(frames_dir, filename)
        if not os.path.isfile(image_path): continue
        embeddings = extract_faces_from_frame(image_path)

        for embedding in embeddings:
            try:
                scaled = scaler.transform([embedding])
                pred_id = model.predict(scaled)[0]
                label = le.inverse_transform([pred_id])[0].lower()
                if (label in leader_classes) or (label in leader_classes_lower):
                    original_name = leader_classes[leader_classes_lower.index(label)]
                    detected[f'leader_{original_name}'] = 1
            except Exception as e:
                print(f"Classification failed on {filename}: {e}")

    return detected
