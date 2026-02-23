import subprocess
import os

# Paths
model_weights_25 = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_incomplete/yolov5m_D3_incomplete_25/weights/best.pt'
model_weights_50 = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_incomplete/yolov5m_D3_incomplete_50/weights/best.pt'
model_weights_75 = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_incomplete/yolov5m_D3_incomplete_75/weights/best.pt'



data_yaml_25 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_25/data.yaml'
data_yaml_50 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_50/data.yaml'
data_yaml_75 = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/incomplete_data/train_75/data.yaml'





project_name = 'yolov5_experiments_incomplete'
run_name_25 = 'yolov5m_D3_incomplete_results_25'
run_name_50 = 'yolov5m_D3_incomplete_results_50'
run_name_75 = 'yolov5m_D3_incomplete_results_75'

# Evaluation command
command = (
    f"python val.py "
    f"--weights {model_weights_25} "
    f"--data {data_yaml_25} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name_25} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name_25}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name_25))


command = (
    f"python val.py "
    f"--weights {model_weights_50} "
    f"--data {data_yaml_50} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name_50} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name_50}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name_50))




command = (
    f"python val.py "
    f"--weights {model_weights_75} "
    f"--data {data_yaml_75} "
    f"--task test "
    f"--img 640 "
    f"--project {project_name} "
    f"--name {run_name_75} "
    f"--exist-ok"
)

print(f"🚀 Starting YOLOv5 test evaluation on real data: {run_name_75}")
subprocess.run(command, shell=True, check=True)
print("✅ Evaluation completed. Check results in:", os.path.join(project_name, run_name_75))

