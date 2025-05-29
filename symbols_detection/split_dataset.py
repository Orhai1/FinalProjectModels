import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Input paths
base_dir = "labeled_data/v3"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
classes_path = os.path.join(base_dir, "classes.txt")

# Output structure
split_base = "labeled_dataset_v3"
train_img_dir = os.path.join(split_base, "images/train")
val_img_dir = os.path.join(split_base, "images/val")
test_img_dir = os.path.join(split_base, "images/test")
train_lbl_dir = os.path.join(split_base, "labels/train")
val_lbl_dir = os.path.join(split_base, "labels/val")
test_lbl_dir = os.path.join(split_base, "labels/test")

for d in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Match images with labels
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
paired_images = []

for img_file in image_files:
    base = os.path.splitext(img_file)[0]
    label_file = base + ".txt"
    if os.path.exists(os.path.join(labels_dir, label_file)):
        paired_images.append((img_file, label_file))

# Optional: load dominant class per image for stratify (best-effort)
dominant_classes = []
for img, lbl in paired_images:
    with open(os.path.join(labels_dir, lbl)) as f:
        lines = f.readlines()
    classes = [int(float(l.split()[0])) for l in lines if l.strip()]
    dominant_classes.append(min(classes) if classes else -1)

# Split train/val/test: 70/20/10
train_val, test = train_test_split(paired_images, test_size=0.1, stratify=dominant_classes, random_state=42)
train_val_classes = [dominant_classes[paired_images.index(i)] for i in train_val]
train, val = train_test_split(train_val, test_size=0.2222, stratify=train_val_classes, random_state=42)  # 0.2222 * 0.9 ≈ 0.2

# Copy files
def copy_split(data, img_out, lbl_out):
    for img, lbl in data:
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_out, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(lbl_out, lbl))

copy_split(train, train_img_dir, train_lbl_dir)
copy_split(val, val_img_dir, val_lbl_dir)
copy_split(test, test_img_dir, test_lbl_dir)

# Load class names
with open(classes_path) as f:
    class_names = [line.strip() for line in f if line.strip()]

# Generate YAML
yaml_path = os.path.join(split_base, "data.yaml")
with open(yaml_path, 'w') as f:
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write(f"test: images/test\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print("Dataset split complete with train/val/test and data.yaml generated at:", yaml_path)
