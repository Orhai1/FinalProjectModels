import os
import albumentations as A
import cv2
from matplotlib import pyplot as plt

# Define paths
input_images_dir = 'labeled_data/v2/images'
input_labels_dir = 'labeled_data/v2/labels'
output_images_dir = 'augmented/images'
output_labels_dir = 'augmented/labels'

# Create output folders
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Load YOLO-format labels (normalized)
def load_yolo_labels(label_path, img_width, img_height):
    bboxes = []
    class_ids = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(float(parts[0]))  # fix for float class_ids
            x_center, y_center, w, h = map(float, parts[1:])
            x_min = (x_center - w / 2) * img_width
            x_max = (x_center + w / 2) * img_width
            y_min = (y_center - h / 2) * img_height
            y_max = (y_center + h / 2) * img_height
            bboxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(class_id)
    return bboxes, class_ids

# Save labels back in YOLO format
def save_yolo_labels(file_path, bboxes, class_ids, img_width, img_height):
    with open(file_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_ids):
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Visual check helper
def show_image_with_boxes(image, bboxes):
    img_copy = image.copy()
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Define safe symbol-aware augmentations
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.05), rotate=(-20, 20), fit_output=True, p=0.6),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.3),
    A.RandomResizedCrop(size=(224,224), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.4),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], check_each_transform=True))

# Loop through each image
for img_file in os.listdir(input_images_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_images_dir, img_file)
    label_path = os.path.join(input_labels_dir, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    if image is None or not os.path.exists(label_path):
        continue

    h, w = image.shape[:2]
    bboxes, class_ids = load_yolo_labels(label_path, w, h)

    for i in range(5):  # create 5 augmentations per image
        try:
            augmented = augment(image=image, bboxes=bboxes, class_labels=class_ids)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_ids = augmented['class_labels']

            # Filter out any zero-area or out-of-bounds boxes
            filtered_boxes = []
            filtered_ids = []
            for box, cid in zip(aug_bboxes, aug_class_ids):
                x_min, y_min, x_max, y_max = box
                if x_max - x_min > 1 and y_max - y_min > 1:
                    filtered_boxes.append([max(0, x_min), max(0, y_min), min(aug_image.shape[1], x_max), min(aug_image.shape[0], y_max)])
                    filtered_ids.append(cid)

            if not filtered_boxes:
                continue  # skip saving if all boxes were removed

            out_img_name = os.path.splitext(img_file)[0] + f"_aug_{i}.jpg"
            out_img_path = os.path.join(output_images_dir, out_img_name)
            out_label_path = os.path.join(output_labels_dir, os.path.splitext(out_img_name)[0] + '.txt')

            # Visual check
            # show_image_with_boxes(aug_image, filtered_boxes)

            cv2.imwrite(out_img_path, aug_image)
            new_h, new_w = aug_image.shape[:2]
            save_yolo_labels(out_label_path, filtered_boxes, filtered_ids, new_w, new_h)

        except Exception as e:
            print(f"Augmentation failed for {img_file} (aug_{i}): {e}")
