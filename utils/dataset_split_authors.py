import os
import shutil
import pandas as pd

# Function to read CSV file and return list of filenames
def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['image_id'].tolist()

# Source directories containing the original images and annotations
source_images_dir = '/cluster/home/ammaa/Downloads/FracAtlas/images/'
source_annotations_dir = '/cluster/home/ammaa/Downloads/FracAtlas/Annotations/YOLO/'

# Destination directory where the split will be placed
destination_dir = '/cluster/home/ammaa/Downloads/FracAtlas/Split'

# Create destination directories if they don't exist
for folder in ['images/train', 'images/test', 'images/val', 'labels/train', 'labels/test', 'labels/val']:
    os.makedirs(os.path.join(destination_dir, folder), exist_ok=True)

# Read filenames from CSV files
train_files = read_csv('/cluster/home/ammaa/Downloads/FracAtlas/Utilities/Fracture Split/train.csv')
test_files = read_csv('/cluster/home/ammaa/Downloads/FracAtlas/Utilities/Fracture Split/test.csv')
valid_files = read_csv('/cluster/home/ammaa/Downloads/FracAtlas/Utilities/Fracture Split/valid.csv')

# Move files to destination directory based on split
for file in train_files:
    # Move images
    source_image_path = os.path.join(source_images_dir, file)
    destination_image_path = os.path.join(destination_dir, 'images', 'train', file)
    shutil.copy(source_image_path, destination_image_path)
    # Move annotations
    txt_file = file[:-4] + ".txt" 
    source_annotation_path = os.path.join(source_annotations_dir, txt_file)
    destination_annotation_path = os.path.join(destination_dir, 'labels', 'train', txt_file)
    shutil.copy(source_annotation_path, destination_annotation_path)

for file in test_files:
    # Move images
    source_image_path = os.path.join(source_images_dir, file)
    destination_image_path = os.path.join(destination_dir, 'images', 'test', file)
    shutil.copy(source_image_path, destination_image_path)
    # Move annotations
    txt_file = file[:-4] + ".txt"
    source_annotation_path = os.path.join(source_annotations_dir, txt_file)
    destination_annotation_path = os.path.join(destination_dir, 'labels', 'test', txt_file)
    shutil.copy(source_annotation_path, destination_annotation_path)

for file in valid_files:
    # Move images
    source_image_path = os.path.join(source_images_dir, file)
    destination_image_path = os.path.join(destination_dir, 'images', 'val', file)
    shutil.copy(source_image_path, destination_image_path)
    # Move annotations
    txt_file = file[:-4] + ".txt"
    source_annotation_path = os.path.join(source_annotations_dir, txt_file)
    destination_annotation_path = os.path.join(destination_dir, 'labels', 'val', txt_file)
    shutil.copy(source_annotation_path, destination_annotation_path)

print("Split completed successfully.")
