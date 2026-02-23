import subprocess
import os

# Paths and configuration
base_path = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper'
model = 'yolov5m'
dataset = 'D3'

# Training parameters
epochs = 150
patience = 20
imgsz = 640
batch = 32

# File paths
weights = f'{model}.pt'  # Pretrained weights
data_yaml = os.path.join(base_path, dataset, 'data_valid_without_real.yaml')
project_name = 'yolov5_experiment_journal_paper_valid_without_real'
run_name = f'{model}_{dataset}'
resume_weights = os.path.join('runs/train', run_name, 'weights', 'last.pt')

# Construct command
if os.path.exists(resume_weights):
    command = f"python train.py --resume"
    print(f"🔁 Resuming training from last checkpoint: {run_name}")
else:
    command = (
        f"python train.py "
        f"--img {imgsz} --batch {batch} --epochs {epochs} "
        f"--patience {patience} --weights {weights} --data {data_yaml} "
       # f"--hyp data/hyps/custom.yaml "  # 👈 your custom .yaml
        f"--project {project_name} --name {run_name} --exist-ok"
    )
    print(f"🚀 Starting training from scratch: {run_name}")

# Execute
subprocess.run(command, shell=True, check=True)

print("✅ YOLOv5 training  completed.")
