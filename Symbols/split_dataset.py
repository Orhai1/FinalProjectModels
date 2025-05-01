import os
import random
import shutil
from collections import defaultdict

# Paths to the labeled dataset
images_dir = 'augmented/images'
labels_dir = 'augmented/labels'

# Output paths
train_img_dir = 'labeled_dataset_v2/images/train'
train_lbl_dir = 'labeled_dataset_v2/labels/train'
val_img_dir = 'labeled_dataset_v2/images/val'
val_lbl_dir = 'labeled_dataset_v2/labels/val'
test_img_dir = 'labeled_dataset_v2/images/test'
test_lbl_dir = 'labeled_dataset_v2/labels/test'

# Create output folders
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Organize images by class
class_to_files = defaultdict(list)
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue
        first_class = int(float(lines[0].split()[0]))
        image_file = os.path.splitext(label_file)[0] + '.jpg'
        if os.path.exists(os.path.join(images_dir, image_file)):
            class_to_files[first_class].append((image_file, label_file))

# Stratified split per class: 70% train, 20% val, 10% test
train_files = []
val_files = []
test_files = []

for cls, files in class_to_files.items():
    random.shuffle(files)
    n = len(files)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    train_files.extend(files[:n_train])
    val_files.extend(files[n_train:n_train + n_val])
    test_files.extend(files[n_train + n_val:])

def copy_pairs(file_list, target_img_dir, target_lbl_dir):
    for img_file, lbl_file in file_list:
        src_img_path = str(os.path.join(images_dir, img_file))
        dst_img_path = str(os.path.join(target_img_dir, img_file))
        src_lbl_path = str(os.path.join(labels_dir, lbl_file))
        dst_lbl_path = str(os.path.join(target_lbl_dir, lbl_file))

        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_lbl_path, dst_lbl_path)

copy_pairs(train_files, train_img_dir, train_lbl_dir)
copy_pairs(val_files, val_img_dir, val_lbl_dir)
copy_pairs(test_files, test_img_dir, test_lbl_dir)

print(f"Stratified split complete. {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")

# Create detect.yaml for YOLO training
yaml_path = 'labeled_dataset_v2/detect.yaml'
class_names = ["fatah_101",
                "fatah_gun",
                "fatah_intelligence",
                "fatah_mount",
                "fatah_police",
                "fatah_presidential",
                "fatah_youth",
                "hamas_ribbon",
                "hamas_symbol"]

with open(yaml_path, 'w') as f:
    f.write(f"train: {train_img_dir}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write(f"test: {test_img_dir}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print(f"detect.yaml generated at {yaml_path}")
