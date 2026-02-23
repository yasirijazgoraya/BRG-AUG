from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')  # or path to a custom model

# Train
#model.train(
 #   data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/data_incomplete.yaml',
  #  epochs=150,
   # imgsz=640,
    #batch=16,
    #name='yolov8m_D3',
   # project='yolov8_experiments_data_incomplete',
   # exist_ok=True
#)



# Train
#model.train(
#    data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_30p.yaml',
 #   epochs=150,
  #  imgsz=640,
  #  batch=16,
  #  name='yolov8m_D3_30p',
  #  project='yolov8_experiments_sparse',
   # exist_ok=True
#)


# Train
#model.train(
 #   data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_60p.yaml',
  #  epochs=150,
   # imgsz=640,
   # batch=16,
   # name='yolov8m_D3_60p',
  #  project='yolov8_experiments_sparse',
  #  exist_ok=True
#)


# Train
#model.train(
   # data='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_90p.yaml',
   # epochs=150,
   # imgsz=640,
   # batch=16,
   # name='yolov8m_D3_90p',
  #  project='yolov8_experiments_sparse',
 #   exist_ok=True
#)



