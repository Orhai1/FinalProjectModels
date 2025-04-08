import os
import cv2
import albumentations as A

# Define augmentation pipeline
# augment = A.Compose([
#     A.Rotate(limit=15),
#     A.RandomBrightnessContrast(),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
#     A.GaussianBlur(p=0.2),
#     A.HorizontalFlip(p=0.5),
# ])

# augment = A.Compose([
#     A.Rotate(limit=20),  # slightly more rotation
#     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
#     A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.7),
#     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#     A.ToGray(p=0.1),  # occasional grayscale
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
#     A.GaussianBlur(blur_limit=3, p=0.3),
#     A.Affine(translate_percent=0.05, scale=1.0, rotate=10, p=0.5),
#     A.HorizontalFlip(p=0.5),
# ])

augment = A.Compose([
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
    A.ToGray(p=0.1),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
])


# Base directories
input_base = "original_photos"
output_base = "augmented_photos3"

# Walk through all directories and files in the original_photos directory
for root, dirs, files in os.walk(input_base):
    for file in files:

        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, input_base)
        output_dir = os.path.join(output_base, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read: {input_path}")
            continue

        # Generate 5 augmented copies
        for j in range(5):
            augmented = augment(image=image)['image']
            filename_no_ext = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"{filename_no_ext}_aug_{j}.jpg")
            cv2.imwrite(output_path, augmented)