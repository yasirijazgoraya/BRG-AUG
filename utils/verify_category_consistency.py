import json
import os

# Paths to your annotation files
base_path = "/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3"
json_files = {
    "train": os.path.join(base_path, "train/train.json"),
    "valid": os.path.join(base_path, "valid/valid.json"),
    "test": os.path.join(base_path, "test/test.json"),
}

# Load categories from each file
category_sets = {}
for split, path in json_files.items():
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            categories = data.get("categories", [])
            category_sets[split] = sorted(categories, key=lambda c: c["id"])
    except Exception as e:
        print(f"❌ Error loading {split} from {path}: {e}")
        category_sets[split] = []

# Compare category lists
all_ok = True
reference = category_sets["train"]
for split, categories in category_sets.items():
    if categories != reference:
        all_ok = False
        print(f"❌ Inconsistency in '{split}':")
        print(f"Expected: {reference}")
        print(f"Found:    {categories}")
    else:
        print(f"✅ Categories in '{split}' match.")

# Final summary
if all_ok:
    print("\n🎉 All annotation files have consistent category definitions.")
else:
    print("\n⚠️ Found inconsistencies in category definitions.")
