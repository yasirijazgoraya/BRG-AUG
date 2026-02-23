import os
import json
from PIL import Image

# Paths
image_dir = '90p/images'
label_dir = '90p/labels'
output_json = '90p/90p.json'

# Class list
class_names = ['Normal_LS', 'Abnormal_LS']

# COCO structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "Normal_LS", "supercategory": "none"},
        {"id": 1, "name": "Abnormal_LS", "supercategory": "none"}
    ]
}

# Flags
check_unlabeled_images = True  # Set to False to skip warning for unmatched images

annotation_id = 1
image_id = 1

# List of available images (without extension)
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# Track matched images
matched_images = set()

# Process labels
for label_file in label_files:
    base_name = os.path.splitext(label_file)[0]

    if base_name not in image_files:
        print(f"❌ No image found for label: {label_file}")
        continue

    image_file = image_files[base_name]
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)

    matched_images.add(base_name)

    # Get image size
    with Image.open(image_path) as img:
        width, height = img.size

    # Add image info
    coco_output["images"].append({
        "file_name": image_file,
        "height": height,
        "width": width,
        "id": image_id
    })

    # Read label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Malformed label in {label_file}: {line.strip()}")
            continue

        class_id, x_center, y_center, w, h = map(float, parts)

        # Convert to COCO bbox
        x = (x_center - w / 2) * width
        y = (y_center - h / 2) * height
        bbox_width = w * width
        bbox_height = h * height

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [x, y, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

# Optional: Warn about images without labels
if check_unlabeled_images:
    for base_name, image_file in image_files.items():
        if base_name not in matched_images:
            print(f"⚠️ No label file found for image: {image_file}")

# Save JSON
with open(output_json, 'w') as f:
    json.dump(coco_output, f, indent=4)

print(f"\n✅ COCO JSON saved to: {output_json}")

