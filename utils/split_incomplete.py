import os
import shutil
import random

# === CONFIG ===
images_dir = "train/images"             # Original images
labels_dir = "train/labels"             # Original labels

output_images_dir = "output/images"
output_labels_dir = "output/labels"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# === INIT ===
class_0_only_files = set()
class_1_files = set()
total_class_0_annotations = 0
total_class_1_annotations = 0

# === STEP 1: Parse all label files ===
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        continue

    class_ids = [int(line.strip().split()[0]) for line in lines]
    class_set = set(class_ids)

    if class_set == {0}:
        class_0_only_files.add(label_file)
        total_class_0_annotations += len(class_ids)

    if 1 in class_set:
        class_1_files.add(label_file)
        total_class_1_annotations += class_ids.count(1)

# === STEP 2: Stats ===
print("📊 Dataset Summary:")
print(f"  Total NORMAL samples (class 0 only): {len(class_0_only_files)}")
print(f"  Total ABNORMAL samples (contains class 1): {len(class_1_files)}")
print(f"  Class 0 annotations: {total_class_0_annotations}")
print(f"  Class 1 annotations: {total_class_1_annotations}")

# === STEP 3: Select 50% of normal samples (class 0 only) ===
#selected_class_0 = set(random.sample(list(class_0_only_files), int(len(class_0_only_files) * 0.25)))
#selected_class_0 = set(random.sample(list(class_0_only_files), len(class_0_only_files) // 2))
selected_class_0 = set(random.sample(list(class_0_only_files), int(len(class_0_only_files) * 0.75)))


# === STEP 4: Ensure no overlaps — abnormal wins if overlapping ===
final_normal_files = selected_class_0 - class_1_files
final_abnormal_files = class_1_files

print(f"\n✅ Final selection:")
print(f"  Normal (only class 0): {len(final_normal_files)}")
print(f"  Abnormal (has class 1): {len(final_abnormal_files)}")

# === STEP 5: Copy selected files ===
def copy_pair(label_file):
    base_name = os.path.splitext(label_file)[0]
    label_src = os.path.join(labels_dir, label_file)
    image_candidates = [base_name + ext for ext in ['.jpg', '.png', '.jpeg']]
    
    for img_file in image_candidates:
        image_src = os.path.join(images_dir, img_file)
        if os.path.exists(image_src):
            shutil.copy(image_src, os.path.join(output_images_dir, os.path.basename(image_src)))
            shutil.copy(label_src, os.path.join(output_labels_dir, label_file))
            return
    print(f"⚠️ Image not found for: {label_file}")

# Copy normal class
for label_file in final_normal_files:
    copy_pair(label_file)

# Copy abnormal class
for label_file in final_abnormal_files:
    copy_pair(label_file)

# === FINAL SUMMARY ===
print("\n🚀 Done. Final dataset:")
print(f"  Images copied: {len(os.listdir(output_images_dir))}")
print(f"  Labels copied: {len(os.listdir(output_labels_dir))}")

