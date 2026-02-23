import subprocess
import os

# Paths

model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_0p/weights/best.pt'
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_train_data/0p/data.yaml'

project_name = 'yolov5_experiments_sparse'
run_name = 'yolov5m_D3_result_0p'

# Evaluation command
command = (
    f"python val.py "
    f"--weights {model_weights} "
    f"--data {data_yaml} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name))











model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_30p/weights/best.pt'
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_train_data/30p/data.yaml'

project_name = 'yolov5_experiments_sparse'
run_name = 'yolov5m_D3_result_30p'

# Evaluation command
command = (
    f"python val.py "
    f"--weights {model_weights} "
    f"--data {data_yaml} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name))













model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_60p/weights/best.pt'
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_train_data/60p/data.yaml'

project_name = 'yolov5_experiments_sparse'
run_name = 'yolov5m_D3_result_60p'

# Evaluation command
command = (
    f"python val.py "
    f"--weights {model_weights} "
    f"--data {data_yaml} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name))







model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_90p/weights/best.pt'
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_train_data/90p/data.yaml'

project_name = 'yolov5_experiments_sparse'
run_name = 'yolov5m_D3_result_90p'

# Evaluation command
command = (
    f"python val.py "
    f"--weights {model_weights} "
    f"--data {data_yaml} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name))




