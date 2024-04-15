import os
import random
import shutil

# Define source and destination directories
source_image_dir = "/cluster/home/ammaa/Downloads/FracAtlas/classes images/"
source_annotation_dir = "/cluster/home/ammaa/Downloads/FracAtlas/Annotations/YOLO/"
destination_dir = "/cluster/home/ammaa/Downloads/FracAtlas/Split/"

# Define train, validation, and test ratios
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

# Create destination directories
os.makedirs(os.path.join(destination_dir, "images", "train", "Fractured-Aug"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "train", "Non-fractured"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "val", "Fractured-Aug"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "val", "Non-fractured"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "test", "Fractured-Aug"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "test", "Non-fractured"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "labels", "test"), exist_ok=True)

# List all images and annotations
images = os.listdir(os.path.join(source_image_dir, "Fractured-Aug")) + os.listdir(os.path.join(source_image_dir, "Non_fractured"))
annotations = os.listdir(source_annotation_dir)

# Shuffle images to ensure randomness in the split
random.shuffle(images)

# Calculate split indices
total_images = len(images)
train_split_idx = int(total_images * train_ratio)
val_split_idx = int(total_images * (train_ratio + val_ratio))

# Split images and annotations
train_images = images[:train_split_idx]
val_images = images[train_split_idx:val_split_idx]
test_images = images[val_split_idx:]

# Move images to respective directories
for img in train_images:
    if img in os.listdir(os.path.join(source_image_dir, "Fractured-Aug")):
        shutil.copy(os.path.join(source_image_dir, "Fractured-Aug", img), os.path.join(destination_dir, "images", "train", "Fractured-Aug"))
    else:
        shutil.copy(os.path.join(source_image_dir, "Non_fractured", img), os.path.join(destination_dir, "images", "train", "Non-fractured"))

for img in val_images:
    if img in os.listdir(os.path.join(source_image_dir, "Fractured-Aug")):
        shutil.copy(os.path.join(source_image_dir, "Fractured-Aug", img), os.path.join(destination_dir, "images", "val", "Fractured-Aug"))
    else:
        shutil.copy(os.path.join(source_image_dir, "Non_fractured", img), os.path.join(destination_dir, "images", "val", "Non-fractured"))

for img in test_images:
    if img in os.listdir(os.path.join(source_image_dir, "Fractured-Aug")):
        shutil.copy(os.path.join(source_image_dir, "Fractured-Aug", img), os.path.join(destination_dir, "images", "test", "Fractured-Aug"))
    else:
        shutil.copy(os.path.join(source_image_dir, "Non_fractured", img), os.path.join(destination_dir, "images", "test", "Non-fractured"))

# Move annotations to respective directories
for ann in annotations:
    if ann[:-4] + ".jpg" in train_images:
        shutil.copy(os.path.join(source_annotation_dir, ann), os.path.join(destination_dir, "labels", "train"))
    elif ann[:-4] + ".jpg" in val_images:
        shutil.copy(os.path.join(source_annotation_dir, ann), os.path.join(destination_dir, "labels", "val"))
    elif ann[:-4] + ".jpg" in test_images:
        shutil.copy(os.path.join(source_annotation_dir, ann), os.path.join(destination_dir, "labels", "test"))
