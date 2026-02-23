import os
import shutil
import random

def create_sparse_datasets(train_dir, output_base_dir, percentages=[0.3, 0.6, 0.9]):
    image_dir = os.path.join(train_dir, "images")
    label_dir = os.path.join(train_dir, "labels")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # for consistent splits
    total_images = len(image_files)

    print(f"Total images found: {total_images}")

    for pct in percentages:
        num_samples = int(total_images * pct)
        selected_files = random.sample(image_files, num_samples)
        split_name = f"{int(pct * 100)}p"

        out_image_dir = os.path.join(output_base_dir, split_name, "images")
        out_label_dir = os.path.join(output_base_dir, split_name, "labels")
        os.makedirs(out_image_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        for img_file in selected_files:
            label_file = img_file.rsplit('.', 1)[0] + ".txt"

            shutil.copy(os.path.join(image_dir, img_file), os.path.join(out_image_dir, img_file))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(out_label_dir, label_file))

        print(f"{split_name} set: Copied {num_samples} images and labels.")

# Example usage
train_directory = "train/"
output_directory = "Sparse_training"
create_sparse_datasets(train_directory, output_directory)

