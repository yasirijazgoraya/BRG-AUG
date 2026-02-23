import json
import os

# Base path
data_root = 'sparse_train_data/'

splits = {
    '30p': data_root + '30p/30p.json',
    '60p': data_root + '60p/60p.json',
    '90p': data_root + '90p/90p.json'
}

for split, path in splits.items():
    with open(path, 'r') as f:
        data = json.load(f)

    # Update categories
    new_categories = [
        {"id": 1, "name": "Normal_LS"},
        {"id": 2, "name": "Abnormal_LS"}
    ]
    data['categories'] = new_categories

    # Update annotation category_ids
    for ann in data['annotations']:
        if ann['category_id'] == 0:
            ann['category_id'] = 1
        elif ann['category_id'] == 1:
            ann['category_id'] = 2

    # Save updated file (backup original just in case)
    backup_path = path.replace('.json', '_original.json')
    os.rename(path, backup_path)
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f"Updated {split} → category_ids shifted from 0-based to 1-based.")
