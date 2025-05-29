import os
import shutil
import random
from collections import defaultdict

# Input paths
SOURCE_IMAGES = 'labeled_data/v3/images'
SOURCE_LABELS = 'labeled_data/v3/labels'
DEST_ROOT = 'labeled_dataset_v3_updated'

# Create output folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(DEST_ROOT, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, 'labels', split), exist_ok=True)

# Helper to move files
def move(img_file, split):
    label_file = os.path.splitext(img_file)[0] + '.txt'
    img_src = os.path.join(SOURCE_IMAGES, img_file)
    label_src = os.path.join(SOURCE_LABELS, label_file)
    img_dst = os.path.join(DEST_ROOT, 'images', split, img_file)
    label_dst = os.path.join(DEST_ROOT, 'labels', split, label_file)

    shutil.copy(img_src, img_dst)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"Label not found for {img_file}")

# 1. Group frame images by video ID
video_groups = defaultdict(list)
non_frame_images = []

for img_file in os.listdir(SOURCE_IMAGES):
    if not img_file.lower().endswith(('.jpg', '.png')):
        continue
    if '_frame_' in img_file:
        video_id = img_file.split('_frame')[0]
        video_groups[video_id].append(img_file)
    else:
        non_frame_images.append(img_file)

# 2. Split frame groups by video ID (80/20)
video_ids = list(video_groups.keys())
random.shuffle(video_ids)
split_index = int(len(video_ids) * 0.8)
train_videos = set(video_ids[:split_index])
val_videos = set(video_ids[split_index:])

# 3. Split non-frame images (80/20)
random.shuffle(non_frame_images)
split_index = int(len(non_frame_images) * 0.8)
train_images = non_frame_images[:split_index]
val_images = non_frame_images[split_index:]

# 4. Move frame images
for vid in train_videos:
    for img in video_groups[vid]:
        move(img, 'train')

for vid in val_videos:
    for img in video_groups[vid]:
        move(img, 'val')

# 5. Move non-frame images
for img in train_images:
    move(img, 'train')

for img in val_images:
    move(img, 'val')

print("Dataset split complete. Frame groups kept together. Others randomly assigned.")
