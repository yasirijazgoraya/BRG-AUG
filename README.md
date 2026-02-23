# BRG-AUG: Bridging the Reality Gap Between Synthetic and Real Samples for Semiconductor Load Station Visual Inspection

[![Paper](https://img.shields.io/badge/Paper-Neural%20Computing%20%26%20Applications-blue)](https://doi.org/INSERT_DOI)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![Blender](https://img.shields.io/badge/Blender-3.x%2B-orange)](https://www.blender.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.x-red)](https://github.com/open-mmlab/mmdetection)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

**BRG-AUG** is a CAD-driven synthetic data generation pipeline that integrates geometric modeling, realistic material simulation, and high-dynamic-range (HDR) environments to produce domain-randomized synthetic samples for semiconductor load station visual inspection.

The framework enables:
- Construction of semantic features for operational and non-operational load station states
- Automated dataset annotation without manual labeling
- Simulation of diverse operational scenarios (occlusion, misalignment, etc.)
- Training and evaluation of state-of-the-art object detectors

<p align="center">
  <img src="figures/pipeline.png" alt="BRG-AUG Pipeline" width="800"/>
</p>

## Key Results

- Achieves **96.3% mAP@50** using YOLOv5 trained exclusively on synthetic data and evaluated on real industrial samples
- **98.7% mAP@50** with Cascade R-CNN under baseline training
- Demonstrates significant performance gains under **sparse** and **incomplete** training conditions

## Repository Structure

```
BRG-AUG/
│
├── rendering/                             # Blender-based synthetic data generation
│   ├── render_normal.py                   # Render operational (normal) load station samples
│   ├── render_misaligned.py               # Render misaligned load station samples
│   └── render_occlusion.py               # Render occluded samples (geometric & photorealistic)
│
├── preprocessing/                         # Data preprocessing pipeline
│   ├── resize.py                          # Image resizing with letterboxing + YOLO label adjustment
│   └── augmentation.py                    # Color-based augmentation (3x dataset generation)
│
├── training/                              # Model training & testing scripts
│   ├── yolov5/                            # YOLOv5 training and evaluation
│   │   ├── train_baseline_yolov5.py       # Baseline training on BRG-AUG synthetic data
│   │   ├── train_baseline_tsne.py         # Baseline training for t-SNE feature analysis
│   │   ├── train_incomplete.py            # Training under incomplete data scenarios
│   │   ├── train_sparse.py                # Training under sparse data scenarios (0–90%)
│   │   ├── test_baseline.py               # Test baseline model on real data
│   │   ├── test_incomplete.py             # Test incomplete data models
│   │   ├── test_sparse.py                 # Test sparse data models
│   │   └── validate_baseline.py           # Validate trained model on real test data
│   ├── yolov8/                            # YOLOv8 training and evaluation
│   │   ├── train_baseline.py              # Baseline training
│   │   ├── train_incomplete.py            # Training under incomplete data scenarios
│   │   ├── train_sparse.py                # Training under sparse data scenarios
│   │   ├── test_baseline.py               # Test baseline model
│   │   ├── test_incomplete.py             # Test incomplete data models
│   │   └── test_sparse.py                 # Test sparse data models
│   └── mmdetection/                       # MMDetection config files
│       ├── atss_custom.py                 # ATSS detector
│       ├── cascade_rcnn_custom.py         # Cascade R-CNN detector
│       ├── deformable_detr_custom.py      # Deformable DETR detector
│       ├── dino_custom.py                 # DINO detector
│       ├── faster_rcnn_custom.py          # Faster R-CNN detector
│       ├── fcos_custom.py                 # FCOS detector
│       ├── retinanet_custom.py            # RetinaNet detector
│       ├── rtmdet_custom.py               # RTMDet detector
│       ├── sparse_rcnn_custom.py          # Sparse R-CNN detector
│       ├── yolox_custom.py                # YOLOX detector
│       └── yolox_tta.py                   # YOLOX test-time augmentation
│
├── evaluation/                            # Evaluation and analysis scripts
│   ├── tsne_analysis.py                   # t-SNE feature space visualization
│   └── gradcam_analysis.py                # Grad-CAM spatial attention analysis
│
├── utils/                                 # Dataset utilities
│   ├── yolo2json.py                       # Convert YOLO annotations to COCO JSON
│   ├── check_cat_ids.py                   # Verify category IDs in COCO JSON
│   ├── update_ids.py                      # Update category IDs (0-based → 1-based)
│   ├── verify_category_consistency.py     # Check consistency across train/valid/test splits
│   ├── sparse_data.py                     # Create sparse training subsets (30%, 60%, 90%)
│   ├── split_incomplete.py                # Generate incomplete datasets (normal class sampling)
│   └── remove_augmented.py                # Remove augmented image/label pairs
│
├── figures/                               # Pipeline and result figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yasirijazgoraya/BRG-AUG.git
cd BRG-AUG

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Requirements

**For rendering (Blender scripts):**
- [Blender](https://www.blender.org/) 3.x+
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) (`pip install blenderproc`)
- HDRI environment maps (industrial lighting)

**For MMDetection-based detectors:**
- [MMDetection](https://github.com/open-mmlab/mmdetection) 3.x
- [MMCV](https://github.com/open-mmlab/mmcv) 2.x

**For YOLOv5 / YOLOv8:**
- [Ultralytics](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)

## Usage

### 1. Synthetic Data Generation (Blender)

The rendering scripts generate synthetic load station samples using BlenderProc with HDRI lighting and domain randomization.

**Render normal (operational) samples:**
```bash
blenderproc run rendering/render_normal.py
```

**Render misaligned samples:**
```bash
blenderproc run rendering/render_misaligned.py
```

**Render occluded samples (geometric & photorealistic):**
```bash
blenderproc run rendering/render_occlusion.py
```

> **Note:** Update the `BLEND`, `HDRI_DIR`, and `OUT_DIR` paths in each script to match your local setup. The scripts generate images, YOLO-format labels, and a CSV shot log.

### 2. Data Preprocessing

**Resize images to 640×640 with letterboxing:**
```bash
python preprocessing/resize.py
```

**Apply color-based augmentation (3x dataset):**
```bash
python preprocessing/augmentation.py
```

### 3. Annotation Conversion

**Convert YOLO labels to COCO JSON format:**
```bash
python utils/yolo2json.py
```

### 4. Training

**YOLOv5 — Baseline training:**
```bash
python training/yolov5/train_baseline_yolov5.py
```

**YOLOv8 — Baseline training:**
```bash
python training/yolov8/train_baseline.py
```

**Sparse data experiments (0%, 30%, 60%, 90% synthetic):**
```bash
python training/yolov5/train_sparse.py
python training/yolov8/train_sparse.py
```

**Incomplete data experiments (25%, 50%, 75% normal class):**
```bash
python training/yolov5/train_incomplete.py
python training/yolov8/train_incomplete.py
```

**Testing on real data:**
```bash
python training/yolov5/test_baseline.py
python training/yolov8/test_baseline.py
```

**MMDetection-based detectors:**
```bash
python tools/train.py training/mmdetection/faster_rcnn_custom.py
python tools/train.py training/mmdetection/cascade_rcnn_custom.py
python tools/train.py training/mmdetection/fcos_custom.py
python tools/train.py training/mmdetection/atss_custom.py
python tools/train.py training/mmdetection/dino_custom.py
python tools/train.py training/mmdetection/yolox_custom.py
python tools/train.py training/mmdetection/rtmdet_custom.py
python tools/train.py training/mmdetection/sparse_rcnn_custom.py
python tools/train.py training/mmdetection/deformable_detr_custom.py
python tools/train.py training/mmdetection/retinanet_custom.py
```

> **Note:** Update `data_root` in config files and training scripts to match your local dataset paths.

### 5. Evaluation & Analysis

**t-SNE feature space visualization:**
```bash
python evaluation/tsne_analysis.py
```
Extracts features from YOLOv5 SPPF layer, generates t-SNE plots comparing synthetic vs. real domain distributions.

**Grad-CAM spatial attention analysis:**
```bash
python evaluation/gradcam_analysis.py
```
Generates attention heatmaps showing where the model focuses on synthetic and real samples.

### 6. Dataset Preparation for Experiments

**Create sparse training subsets (30%, 60%, 90%):**
```bash
python utils/sparse_data.py
```

**Create incomplete datasets (reduced normal class):**
```bash
python utils/split_incomplete.py
```

### 7. Verification Utilities

```bash
python utils/check_cat_ids.py                   # Check category IDs
python utils/verify_category_consistency.py      # Verify consistency across splits
python utils/update_ids.py                       # Update 0-based → 1-based IDs
```

## Dataset Configurations

The experiments use three dataset configurations:

| Dataset | Description | Classes | Normal Boxes | Abnormal Boxes |
|---------|-------------|---------|--------------|----------------|
| **D0** | Un-augmented synthetic data | Normal_LS, Abnormal_LS | 3,038 | 3,061 |
| **D1** | Augmented with geometric occlusions | Normal_LS, Abnormal_LS | 3,038 | 3,061 |
| **D2** | Augmented with photorealistic occlusions | Normal_LS, Abnormal_LS | 3,038 | 3,061 |

## Training Environment

| Component       | Specification                    |
|-----------------|----------------------------------|
| GPU             | NVIDIA GeForce RTX 4080 (16 GB) |
| CUDA            | 12.4                             |
| cuDNN           | 9.1                              |
| Python          | 3.8.20                           |
| PyTorch         | 2.4.1                            |
| TorchVision     | 0.20.0                           |

## Baseline Results

| Detector       | Stage     | Anchor Type  | Transformer | mAP@50 | mAP@50-95 |
|----------------|-----------|--------------|:-----------:|--------|-----------|
| Cascade R-CNN  | Two-stage | Anchor-based | ✗ | **98.7** | 68.2 |
| ATSS           | One-stage | Anchor-based | ✗ | 97.9 | 49.0 |
| YOLOv8         | One-stage | Anchor-free  | ✗ | 97.7 | 68.2 |
| Faster R-CNN   | Two-stage | Anchor-based | ✗ | 97.7 | 58.1 |
| YOLOv5         | One-stage | Anchor-based | ✗ | 96.3 | **73.3** |
| YOLOX          | One-stage | Anchor-free  | ✗ | 95.6 | 57.7 |
| Sparse R-CNN   | Two-stage | Anchor-free  | ✓ | 70.6 | 30.3 |
| DINO           | Two-stage | Anchor-free  | ✓ | 62.1 | 27.1 |
| RTMDet         | One-stage | Anchor-free  | ✓ | 59.3 | 22.8 |
| FCOS           | One-stage | Anchor-free  | ✗ | 49.5 | 11.3 |

All models were trained on BRG-AUG synthetic data and evaluated on unseen real industrial samples.

## Dataset Availability

The datasets analyzed during the current study are proprietary to our project partner, Seagate Technology, and were used under license for the research presented. Due to the commercial sensitivity and confidentiality agreements governing these data, it is not publicly available in a repository. However, anonymized metadata or specific data-related inquiries may be available from the corresponding author upon reasonable request and with the express written permission of Seagate Technology.

## Citation

If you find this work useful, please cite:

```bibtex
@article{ijaz2026brgaug,
  title={{BRG-AUG}: Bridging the Reality Gap Between Synthetic and Real Samples for Semiconductor Load Station Visual Inspection},
  author={Ijaz, Yasir and Coleman, Sonya and Kerr, Dermot and Siddique, Nazmul and McAteer, Cormac and Baker, Bryan and Nguyen, Khoi},
  journal={Neural Computing and Applications},
  year={2026},
  publisher={Springer}
}
```

## Authors

- **Yasir Ijaz** — Ulster University ([Ijaz-Y@ulster.ac.uk](mailto:Ijaz-Y@ulster.ac.uk))
- **Sonya Coleman** — Ulster University
- **Dermot Kerr** — Ulster University
- **Nazmul Siddique** — Ulster University
- **Cormac McAteer** — Seagate Technology
- **Bryan Baker** — Seagate Technology
- **Khoi Nguyen** — Seagate Technology

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
