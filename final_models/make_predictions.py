import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model and features
model = joblib.load("models/clip_early_fusion/LogisticReg_best.joblib")
X, y, ids = np.load("data/new_aux_clip_features.npz", allow_pickle=True).values()

# Make predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

# Get the final step from the pipeline
final_step = model.steps[-1][1]

if hasattr(final_step, "le") and hasattr(final_step.le, "classes_"):
    # Case: it's a LabelEncoderWrapper
    class_labels = final_step.le.classes_
elif hasattr(final_step, "classes_"):
    # Case: it's a normal classifier
    class_labels = final_step.classes_
else:
    raise AttributeError("Could not determine class labels from final estimator.")

# Build results DataFrame
df = pd.DataFrame({
    "video_id": ids,
    "true_label": y,
    "predicted_label": y_pred
})

# Add per-class probabilities
for idx, cls in enumerate(class_labels):
    df[f"prob_{cls}"] = y_proba[:, idx]

# Save to CSV
results_path = "results/predictions/LogReg.csv"
df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")

# Print evaluation summary
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(pd.DataFrame(
    confusion_matrix(y, y_pred, labels=class_labels),
    index=[f"true_{cls}" for cls in class_labels],
    columns=[f"pred_{cls}" for cls in class_labels]
))

# # use this to only present classes that are in the predictions
# present_classes = sorted(np.unique(y))
#
# print("\nFiltered Classification Report:")
# print(classification_report(
#     y, y_pred, labels=present_classes, target_names=present_classes
# ))