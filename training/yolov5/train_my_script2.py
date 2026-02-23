import subprocess
import os

# ✅ Base dataset path
base_path = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/'
model = 'yolov5m.pt'

# ✅ Training configuration
epochs = 150
imgsz = 640
batch = 32
patience=20

# ✅ Dataset variants and run settings
train_jobs = [
    {
        'rel_path': 'data_incomplete.yaml',
        'name': 'yolov5m_D3',
        'project': 'yolov5_experiments_data_incomplete'
    },
    {
        'rel_path': 'sparse_30p.yaml',
        'name': 'yolov5m_D3_30p',
        'project': 'yolov5_experiments_sparse'
    },
    {
        'rel_path': 'sparse_60p.yaml',
        'name': 'yolov5m_D3_60p',
        'project': 'yolov5_experiments_sparse'
    },
    {
        'rel_path': 'sparse_90p.yaml',
        'name': 'yolov5m_D3_90p',
        'project': 'yolov5_experiments_sparse'
    }
]

# ✅ Run training for each dataset
for job in train_jobs:
    full_data_path = os.path.join(base_path, job['rel_path'])

    command = (
        f"python train.py "
        f"--img {imgsz} --batch {batch} --epochs {epochs} "
        f"--weights {model} --data {full_data_path} "
        f"--project {job['project']} --name {job['name']} --exist-ok"
    )

    print(f"🚀 Starting: {job['name']}")
    subprocess.run(command, shell=True, check=True)

print("✅ All YOLOv5 training runs completed.")
