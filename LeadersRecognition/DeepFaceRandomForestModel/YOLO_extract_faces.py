from deepface import DeepFace
import os
import numpy as np
from joblib import load
model      = load("best_model_randomforest.pkl")
scaler   = load("scaler.pkl")
le       = load("label_encoder.pkl")
model_name = "ArcFace"  # or SFace, Facenet512, etc.

# ✅ Test folder path
test_dir = "../test_set"

correct = 0
total = 0

print("\n🚀 Starting multi-face evaluation on test set...\n")

# ✅ Loop over each person in test set
for person in os.listdir(test_dir):
    person_dir = os.path.join(test_dir, person)
    if not os.path.isdir(person_dir): continue
    print(f"👤 Evaluating for label: {person}")

    # ✅ Loop over each image of the person
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        print(f"  🔍 Processing image: {img_path}")

        try:
            # ✅ Detect and extract all face embeddings in image
            faces = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend="retinaface",  # use a good face detector
                enforce_detection=False,
                align=True
            )

            predicted_labels = []

            for face in faces:
                embedding = face["embedding"]
                scaled_emb = scaler.transform([embedding])
                pred_id = model.predict(scaled_emb)[0]
                pred_label = le.inverse_transform([pred_id])[0].lower()
                predicted_labels.append(pred_label)

            print(f"     🧠 Predicted faces: {predicted_labels}")

            # ✅ Check if true label appears in any prediction
            if person.lower() in predicted_labels:
                correct += 1
                print("     ✅ Correct match!")
            else:
                print("     ❌ Incorrect.")

            total += 1

        except Exception as e:
            print(f"  ⚠️ Skipped {img_path} | Reason: {str(e)[:80]}")

# ✅ Final accuracy
accuracy = correct / total if total else 0
print(f"\n🎯 Multi-face Accuracy: {round(accuracy * 100, 2)}% ({correct}/{total} correct)")
