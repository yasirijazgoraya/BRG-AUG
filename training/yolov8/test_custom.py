from ultralytics import YOLO
import os

# Paths and settings
model_weights = 'yolov8_experiment_augmented/yolov8m_D3/weights/best.pt'  # or provide full path if not in current dir
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/data_augmented.yaml'
project_name = 'yolov8_test_results_without_real'
run_name = 'yolov8m_D3_test_eval'
imgsz = 640
batch = 32
device = 0  # or 'cpu'

# Load the trained model (assuming best.pt exists from training)
model = YOLO(model_weights)

# Run evaluation on test set
print(f"🚀 Starting YOLOv8 test evaluation on real data: {run_name}")
metrics = model.val(
    data=data_yaml,
    imgsz=imgsz,
    batch=batch,
    split='test',
    project=project_name,
    name=run_name,
    device=device,
    save=True
)

# Print results summary
print("✅ Evaluation completed.")
print("📊 mAP@0.5:", metrics.box.map50)
print("📊 mAP@0.5:0.95:", metrics.box.map)
