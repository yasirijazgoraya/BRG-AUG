#!/usr/bin/env python3
"""
Baseline Training for t-SNE Feature Analysis
Train YOLOv5m on BRG-AUG synthetic data
"""

import subprocess
import os

# ============== CONFIGURATION ==============
CONFIG = {
    'model': 'yolov5m',
    'data_yaml': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/augmented_train_data/train/data.yaml',
    'epochs': 100,
    'patience': 20,
    'batch': 32,
    'imgsz': 640,
    'project': 'baseline_tsne_experiment',
    'name': 'yolov5m_baseline',
    'device': '0',
}

def main():
    # Paths
    weights = f"../{CONFIG['model']}.pt"  # Pretrained weights
    
    # Check if data.yaml exists
    if not os.path.exists(CONFIG['data_yaml']):
        print(f"ERROR: Data YAML not found: {CONFIG['data_yaml']}")
        return
    
    # Build command
    cmd = (
        f"python ../train.py "
        f"--img {CONFIG['imgsz']} "
        f"--batch {CONFIG['batch']} "
        f"--epochs {CONFIG['epochs']} "
        f"--patience {CONFIG['patience']} "
        f"--weights {weights} "
        f"--data {CONFIG['data_yaml']} "
        f"--project {CONFIG['project']} "
        f"--name {CONFIG['name']} "
        f"--device {CONFIG['device']} "
        f"--exist-ok"
    )
    
    print("=" * 60)
    print("BASELINE TRAINING FOR t-SNE ANALYSIS")
    print("=" * 60)
    print(f"Model: {CONFIG['model']}")
    print(f"Data: {CONFIG['data_yaml']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch: {CONFIG['batch']}")
    print("=" * 60)
    print(f"\nCommand:\n{cmd}\n")
    print("=" * 60)
    
    # Run training
    subprocess.run(cmd, shell=True, check=True)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Weights saved to: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")

if __name__ == '__main__':
    main()
