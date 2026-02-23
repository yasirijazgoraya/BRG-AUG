#!/usr/bin/env python3
"""
Validate Baseline Model on Real Test Data
"""

import subprocess
import os

# ============== CONFIGURATION ==============
CONFIG = {
    'weights': 'baseline_tsne_experiment/yolov5m_baseline/weights/best.pt',
    'data_yaml': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/augmented_train_data/train/data.yaml',
    'imgsz': 640,
    'batch': 32,
    'project': 'baseline_tsne_experiment',
    'name': 'validation_results',
    'device': '0',
}

def main():
    # Check weights exist
    if not os.path.exists(CONFIG['weights']):
        print(f"ERROR: Weights not found: {CONFIG['weights']}")
        print("Please run training first!")
        return
    
    # Build validation command
    cmd = (
        f"python ../val.py "
        f"--weights {CONFIG['weights']} "
        f"--data {CONFIG['data_yaml']} "
        f"--img {CONFIG['imgsz']} "
        f"--batch {CONFIG['batch']} "
        f"--task test "
        f"--project {CONFIG['project']} "
        f"--name {CONFIG['name']} "
        f"--device {CONFIG['device']} "
        f"--save-txt "
        f"--save-conf "
        f"--exist-ok"
    )
    
    print("=" * 60)
    print("VALIDATING BASELINE MODEL ON REAL TEST DATA")
    print("=" * 60)
    print(f"Weights: {CONFIG['weights']}")
    print(f"Data: {CONFIG['data_yaml']}")
    print("=" * 60)
    
    # Run validation
    subprocess.run(cmd, shell=True, check=True)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETED!")
    print("=" * 60)
    print(f"Results saved to: {CONFIG['project']}/{CONFIG['name']}/")

if __name__ == '__main__':
    main()
