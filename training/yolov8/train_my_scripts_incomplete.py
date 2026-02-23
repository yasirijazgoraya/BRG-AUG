from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')  # or path to a custom model

# Train
model.train(
    data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_25/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    name='yolov8m_D3_25',
    project='yolov8_experiments_incomplete',
    exist_ok=True
)








model.train(
    data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_75/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    name='yolov8m_D3_75',
    project='yolov8_experiments_incomplete',
    exist_ok=True
)

