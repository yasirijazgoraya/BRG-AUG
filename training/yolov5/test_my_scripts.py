import subprocess
import os

# Paths
#model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiment_journal_paper/yolov5m_D3/weights/best.pt'
#model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiment_journal_paper_default_hyps_setting/yolov5m_D3/weights/best.pt'

#model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiment_augmented/yolov5m_D3/weights/best.pt'
#model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_data_incomplete/yolov5m_D3/weights/best.pt'
#model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_30p/weights/best.pt'

model_weights = '/home/yasir/yasir_mnt/external3/object_detectors/ultralytics_models/yolov5/yolov5_experiments_sparse/yolov5m_D3_90p/weights/best.pt'






#data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/data.yaml'
#data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/data_augmented.yaml'

#data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/data_incomplete.yaml'
#data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_30p.yaml'
data_yaml = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/sparse_90p.yaml'







project_name = 'yolov5_test_results'
run_name = 'yolov5m_D3_sparse_90p'

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
