import os
from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# CONFIG
model_path = "symbols_recogniton_models/symbols_tm1/keras_model.h5"
labels_path = "symbols_recogniton_models/symbols_tm1/labels.txt"
input_base_dir = "test_set"  # directory containing folders for each test class
csv_path = "symbol_model_tm1_results"  # path to save results
image_size = (224, 224)

# load model
model = load_model(model_path, compile=False)
class_names = [line.strip() for line in open(labels_path)]

# initialize results list
results = []

# process each folder in the input directory
for folder in os.listdir(input_base_dir):
    folder_path = os.path.join(input_base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for image_file in os.listdir(folder_path):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, image_file)
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(image, image_size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized = (image_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized, axis=0)

            # Prediction
            prediction = model.predict(data)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index]

            # Clean the predicted class label
            predicted_class_clean = predicted_class.strip().lower()

            # Extract the base (true) label from the folder
            true_label_raw = folder.strip().lower()
            true_label_base = true_label_raw.split("_")[0]

            # Check if prediction matches the beginning of the true label
            match = predicted_class_clean.startswith(true_label_base)

            results.append({
                "Folder": folder,
                "Image": image_file,
                "Predicted Class": predicted_class,
                "Confidence": confidence,
                "Match": match
            })

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")


# Save to CSV
df = pd.DataFrame(results)
df.to_csv(f"{csv_path}.csv", index=False)

# Group by Folder (true label)
summary = df.groupby("Folder").agg(
    total_images=pd.NamedAgg(column="Match", aggfunc="count"),
    correct_predictions=pd.NamedAgg(column="Match", aggfunc="sum"),
    average_confidence=pd.NamedAgg(column="Confidence", aggfunc="mean")
)

# Calculate accuracy
summary["Accuracy (%)"] = (summary["correct_predictions"] / summary["total_images"]) * 100
summary["Correct / Total"] = summary["correct_predictions"].astype(str) + " / " + summary["total_images"].astype(str)

# Reorder and format
summary = summary.reset_index()[["Folder", "Correct / Total", "Accuracy (%)", "average_confidence"]]
summary.rename(columns={"average_confidence": "Avg Confidence"}, inplace=True)

# Save to file
summary.to_csv(f"{csv_path}_summary.csv", index=False)