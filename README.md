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
final_showcase/
verified_outputs/         # 130  outputs
all_results.csv           # Tabulated quantitative results across all test sets
verified_pass_list.json   # Verified sample registry
inference_pipeline.py         # End-to-end inference and evaluation script
evaluate_metrics.py           # COCO evaluation and precision-recall metrics
requirements.txt              # Project dependencies
.gitignore                    # Standard repository exclusions
README.md                     # Project documentation
```

---

## ðŸ‘¥ Authors & Acknowledgments
- Developed as part of the Senior Capstone Project.
- Model architectures built on PyTorch, TorchVision Mask R-CNN, and Depth Anything V2.
