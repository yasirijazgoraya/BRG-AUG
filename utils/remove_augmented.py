import os
import re

# Paths
images_path = "D0/train/images"
labels_path = "D0/train/labels"

# File extensions
image_exts = {".jpg", ".jpeg", ".png"}
label_ext = ".txt"

# Match only if filename ends with _1 or _2 before extension
aug_pattern = re.compile(r".*_([12])\.(jpg|jpeg|png)$", re.IGNORECASE)

# Loop through images
for filename in os.listdir(images_path):
    base, ext = os.path.splitext(filename)

    if ext.lower() in image_exts and aug_pattern.match(filename):
        # Delete augmented image
        image_file = os.path.join(images_path, filename)
        os.remove(image_file)
        print(f"Deleted image: {filename}")

        # Delete matching label
        label_file = os.path.join(labels_path, base + label_ext)
        if os.path.exists(label_file):
            os.remove(label_file)
            print(f"Deleted label: {os.path.basename(label_file)}")

