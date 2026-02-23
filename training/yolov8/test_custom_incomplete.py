from ultralytics import YOLO
import os

# Paths and settings
model_weights_25 = 'yolov8_experiments_incomplete/yolov8m_D3_25/weights/best.pt'  # or provide full path if not in current dir
data_yaml_25 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_25/data.yaml'
project_name = 'yolov8_experiments_incomplete'
run_name_25 = 'D3_results_25'


imgsz = 640
batch = 32
device = 0  # or 'cpu'

# Load the trained model (assuming best.pt exists from training)
model = YOLO(model_weights_25)

# Run evaluation on test set
print(f"🚀 Starting YOLOv8 test evaluation on real data: {run_name_25}")
metrics = model.val(
    data=data_yaml_25,
    imgsz=imgsz,
    batch=batch,
    split='test',
    project=project_name,
    name=run_name_25,
    device=device,
    save=True
)

# Print results summary
print("✅ Evaluation completed.")
print("📊 mAP@0.5:", metrics.box.map50)
print("📊 mAP@0.5:0.95:", metrics.box.map)






# Paths and settings
model_weights_50 = 'yolov8_experiments_incomplete/yolov8m_D3_50/weights/best.pt'  # or provide full path if not in current dir
data_yaml_50 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_50/data.yaml'
project_name = 'yolov8_experiments_incomplete'
run_name_50 = 'D3_results_50'


imgsz = 640
batch = 32
device = 0  # or 'cpu'

# Load the trained model (assuming best.pt exists from training)
model = YOLO(model_weights_50)

# Run evaluation on test set
print(f"🚀 Starting YOLOv8 test evaluation on real data: {run_name_50}")
metrics = model.val(
    data=data_yaml_50,
    imgsz=imgsz,
    batch=batch,
    split='test',
    project=project_name,
    name=run_name_50,
    device=device,
    save=True
)

# Print results summary
print("✅ Evaluation completed.")
print("📊 mAP@0.5:", metrics.box.map50)
print("📊 mAP@0.5:0.95:", metrics.box.map)





# Paths and settings
model_weights_75 = 'yolov8_experiments_incomplete/yolov8m_D3_75/weights/best.pt'  # or provide full path if not in current dir
data_yaml_75 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_75/data.yaml'
project_name = 'yolov8_experiments_incomplete'
run_name_75 = 'D3_results_75'


imgsz = 640
batch = 32
device = 0  # or 'cpu'

# Load the trained model (assuming best.pt exists from training)
model = YOLO(model_weights_75)

# Run evaluation on test set
print(f"🚀 Starting YOLOv8 test evaluation on real data: {run_name_75}")
metrics = model.val(
    data=data_yaml_75,
    imgsz=imgsz,
    batch=batch,
    split='test',
    project=project_name,
    name=run_name_75,
    device=device,
    save=True
)

# Print results summary
print("✅ Evaluation completed.")
print("📊 mAP@0.5:", metrics.box.map50)
print("📊 mAP@0.5:0.95:", metrics.box.map)


