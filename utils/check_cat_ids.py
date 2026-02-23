import json
from collections import Counter

# Base dataset path
data_root = 'sparse_train_data/'

splits = {
    '30p': data_root + '30p/30p.json',
    '60p': data_root + '60p/60p.json',
    '90p': data_root + '90p/90p.json'
}

for split_name, json_path in splits.items():
    print(f"\n=== {split_name} ===")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Print categories
        print("Categories:")
        for cat in data['categories']:
            print(f"  ID: {cat['id']}  Name: {cat['name']}")

        # Count category_id usage in annotations
        category_ids = [ann['category_id'] for ann in data['annotations']]
        counts = Counter(category_ids)

        print("Category ID usage in annotations:")
        for cat_id, count in counts.items():
            print(f"  ID {cat_id}: {count} boxes")

    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
