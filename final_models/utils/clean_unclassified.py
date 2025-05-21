import numpy as np
from collections import Counter

# ---------- CONFIG ----------
IN_NPZ  = "../data/aux_features.npz"
OUT_NPZ = "../data/aux_features_clean.npz"
BAD_LABELS = {"Uncertain", "N/A", ""}   # add any others you want to skip
# -----------------------------

# Load original arrays
data     = np.load(IN_NPZ)
X, y, ids = data["X"], data["y"], data["video_id"]

# filter: remove explicit “bad” labels
mask_good = ~np.isin(y, list(BAD_LABELS))
X, y, ids = X[mask_good], y[mask_good], ids[mask_good]

# 3. Second filter: remove classes that still have < MIN_PER_CLASS samples
counts = Counter(y)
valid_classes = {cls for cls, cnt in counts.items()}

print("kept", len(y), "samples across", len(valid_classes), "classes")
print("class counts:", Counter(y))

# 4. Save the cleaned feature file
np.savez(OUT_NPZ, X=X, y=y, video_id=ids)
print(f"Cleaned file written → {OUT_NPZ}")
