# This script augments images and their labels that are in train/val format.
import os
import cv2
import albumentations as A
import shutil

# Settings
DATASET_DIR = "labeled_dataset_v3_updated"
OUTPUT_DATASET_DIR = "aug_labeled_dataset_v3_updated"
SPLITS = ["train", "val"]
AUG_TIMES = 3  # Number of augmentations per image

# Define augmentation pipeline
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.05), rotate=(-20, 20), fit_output=True, p=0.6),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], check_each_transform=True))


for split in SPLITS:
    image_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)

    aug_img_dir = os.path.join(OUTPUT_DATASET_DIR, "images", f"{split}")
    aug_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "labels", f"{split}")

    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        # Copy original image and label
        shutil.copy(image_path, os.path.join(aug_img_dir, filename))
        shutil.copy(label_path, os.path.join(aug_lbl_dir, filename.rsplit('.', 1)[0] + ".txt"))

        # Load image and annotations
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        bboxes = []
        class_labels = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

        for i in range(AUG_TIMES):
            augmented = augment(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            aug_name = filename.rsplit('.', 1)[0] + f"_aug{i}.jpg"
            cv2.imwrite(os.path.join(aug_img_dir, aug_name), aug_img)

            with open(os.path.join(aug_lbl_dir, aug_name.rsplit('.', 1)[0] + ".txt"), "w") as f:
                for bbox, cls in zip(aug_bboxes, aug_labels):
                    f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

print("Done augmenting images and copying originals.")
