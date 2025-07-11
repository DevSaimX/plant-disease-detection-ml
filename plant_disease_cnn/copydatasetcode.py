import os
import shutil
import random

dataset_dir = r"D:\ML Projects(TensorFlow)\New Plant Diseases Dataset(Augmented)"
train_dir = r"D:\ML Projects(TensorFlow)\plant_disease_cnn\dataset\train"
test_dir = r"D:\ML Projects(TensorFlow)\plant_disease_cnn\dataset\test"

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Percentage split
split_ratio = 0.8  # 80% train, 20% test

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        # Create class folders inside train and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        images = os.listdir(class_path)
        random.shuffle(images)

        train_count = int(len(images) * split_ratio)
        train_images = images[:train_count]
        test_images = images[train_count:]

        # Copy train images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            if os.path.isfile(src):
                shutil.copy2(src, dst)


        # Copy test images
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

print("Train/Test split and copy done!")
