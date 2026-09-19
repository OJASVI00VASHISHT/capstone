# Building Height Estimation from Street-View Imagery using Reference-Object Calibrated Monocular Vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computer vision system designed to accurately estimate the metric height (in meters) and floor count of residential and commercial buildings from monocular Google Street View panoramas and street-level imagery.

---

##Problem Statement & Key Challenges

Estimating 3D physical building heights from a single 2D monocular image is inherently ill-posed due to scale ambiguity:
1. **Monocular Depth Scale Ambiguity:** Pretrained foundation models (e.g. Depth Anything V2) output affine-invariant *relative disparity*, which lacks absolute metric scaling. Direct linear mappings lead to 2â€“3Ã— height underestimation.
2. **Foreground Occlusions & Wall Poisoning:** Compound walls, security gates, fences, and vehicles in front of buildings create depth discontinuities that contaminate facade depth sampling.
3. **Bounding Box Truncation:** Standard instance segmentation often clips roofs at upper parapets or truncates multi-floor structures.
..
---

##  Proposed Solution: Multi-Model Reference Calibration

To resolve scale ambiguity without expensive LiDAR or stereo sensors, we introduce a **Reference-Object Depth Ratio Calibration Pipeline**:

```

### Mathematical Formulation

Given a target building of pixel height $h_{\text{house}}^{\text{px}}$ at relative depth $d_{\text{house}}$, and a nearby reference object (e.g. car $H_{\text{ref}} = 1.50\text{m}$, person $H_{\text{ref}} = 1.70\text{m}$) of pixel height $h_{\text{ref}}^{\text{px}}$ at relative depth $d_{\text{ref}}$:

$$H_{\text{house}} = \left(\frac{h_{\text{house}}^{\text{px}}}{h_{\text{ref}}^{\text{px}}}\right) \times H_{\text{ref}} \times \left(\frac{d_{\text{ref}}}{d_{\text{house}}}\right)$$

- **Depth Ratio Independence:** Because Depth Anything V2 preserves relative geometric ordering, the depth ratio cancels monocular scaling constants, yielding metric heights without camera baseline calibration.
- **Constrained Ratio Clamping:** Ratios are bounded to $[0.5, 2.0]$ to eliminate perspective distortion outliers.

---

## Key Features

- **Multi-Model Ensemble:** Ensemble of two fine-tuned Mask R-CNN ResNet-50 FPN models for building detection and segmentation.
- **Multi-Class Metric Anchors:** Pretrained COCO anchors for standard metric references:
  - Cars / Sedans ($1.50\text{m}$) & SUVs ($1.70\text{m}$)
  - Pedestrians ($1.70\text{m}$)
  - Buses ($3.20\text{m}$) & Commercial Trucks ($3.50\text{m}$)
  - Motorcycles ($1.10\text{m}$) & Bicycles ($1.00\text{m}$)
- **Depth-Guided Roofline Extension:** Upward and downward vertical boundary refinement scanning along depth gradients to recover truncated upper floors.
- **Top-Clipping Flagging:** Automatically detects and annotates buildings extending beyond the camera frame.
- **Robust Fallback:** Inverse-disparity geometric projection with $95^\circ$ FOV pinhole camera model when reference objects are absent.

---

## Results & Showcase

Across 270 comprehensive test images (August Test Dataset + Validation Sets):
- **Reference Calibration Rate:** 71% of buildings successfully calibrated with in-scene reference anchors.
- **Realistic Height Distribution:** Median predicted height of **8.9m** (~3 floors), matching real-world ground truth for Indian multi-story residential architecture.
- **130 Verified High-Accuracy Showcase Samples** available in [`final_showcase/verified_outputs/`](final_showcase/verified_outputs/).

---

##  Installation & Quick Start

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/OJASVI00VASHISHT/capstone.git
cd building-height-estimation

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Height Estimation on an Image
```bash
python inference_pipeline.py
```

---

##  Project Structure

```
app/                              # Flask web application
  app.py                          # Main entry point (routes, API)
  compliance_engine.py            # Legal compliance evaluator & image annotator
  gps_extractor.py                # Multi-strategy GPS metadata extractor
  zoning_db.py                    # SQLite zoning regulation database interface
  templates/
    index.html                    # Frontend UI

pipeline/                         # Core ML inference engine
  inference_pipeline.py           # BuildingHeightEstimator class (end-to-end)

training/                         # Model training & evaluation scripts
  train_2000.py                   # Train on unified 2000-image dataset
  train_july.py                   # Train on July 400-image dataset
  precision.py                    # Full mAP / precision-recall benchmark
  evaluate_metrics.py             # COCO evaluation scaffold

data/                             # Annotation & dataset JSON files
  annotations.json / annotations_fixed.json
  images2000.json / images2000_fixed.json / images400.json
  dataset_july_*.json
  instances_default.json
  precision_comparison.json / combinations_precision_results.json

hardware/                         # ESP32 IoT integration
  firmware/
    espcode.ino                   # Arduino firmware for ESP32-CAM
  server/
    image_receiver.py             # Flask server to receive images from ESP32
    server.py / newserver2.py     # ESP32 bridge server variants

notebooks/                        # Jupyter notebooks (exploration & analysis)
  main.ipynb, helper.ipynb, helper_july.ipynb
  testing2.ipynb, testingmodel.ipynb
  datasetdesc.ipynb, datasetdesc_july.ipynb

evaluation/                       # Showcase outputs and result registries
  final_showcase/
    all_results.csv               # Quantitative results across all test sets
    verified_pass_list.json       # Verified sample registry
    verified_outputs/             # 130 verified annotated output images

docs/                             # Project reports & documentation
  Project_Report.docx
  August_Test_Height_Estimate_Report.docx
  Manual_Testing_Ensemble_Height_Estimate.docx

sample_images/                    # Demo input images
Connected Building Landscape.csv  # GPS catalog for image filename lookup
requirements.txt                  # Project dependencies
README.md                         # This file
.gitignore
```


---

## ðŸ‘¥ Authors & Acknowledgments
- Developed as part of the Senior Capstone Project.
- Model architectures built on PyTorch, TorchVision Mask R-CNN, and Depth Anything V2.
