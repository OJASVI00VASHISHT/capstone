# Project Context: Building Height Estimation & Illegal Construction Detection

## 1. Project Overview
This project is an end-to-end computer vision system and web application designed to accurately estimate the metric height (in meters) and floor count of residential and commercial buildings from monocular Google Street View panoramas and street-level imagery. 

The primary goal is to **detect illegal constructions** by comparing the estimated building heights against municipal zoning regulations.

### Key Challenges Addressed
1. **Monocular Depth Scale Ambiguity:** Single 2D images lack absolute metric scaling. 
2. **Foreground Occlusions:** Compound walls, gates, fences, and vehicles can contaminate depth sampling.
3. **Bounding Box Truncation:** Standard instance segmentation often clips roofs or truncates structures.

## 2. Technical Solution & Architecture
The system overcomes these challenges without relying on expensive LiDAR or stereo sensors by using a **Reference-Object Depth Ratio Calibration Pipeline**. It leverages the known metric heights of common objects (like cars, pedestrians) to calibrate the depth scale.

### Core Modules
* **`inference_pipeline.py` (The ML Core):** 
  * Loads an ensemble of two fine-tuned Mask R-CNN ResNet-50 FPN models (for building detection) and a standard COCO model (for reference object detection).
  * Uses Hugging Face's `Depth-Anything-V2-Small-hf` for relative depth estimation.
  * Uses a robust roofline and base algorithm to extend truncated bounding boxes.
  * Identifies reference objects (prioritizing ground anchors like vehicles) and applies a mathematical formulation utilizing depth ratios to derive the metric height of the target building.
  * Implements fallback geometric projections (pinhole camera model) if reference objects are absent.
* **`compliance_engine.py` (The Legal Evaluator):** 
  * Computes approximate GPS coordinates for the detected buildings based on the camera's location, distance, and azimuth.
  * Cross-references the estimated building heights with municipal zoning limits (fetched from the database).
  * Flags buildings as **LEGAL**, **WARNING** (near limit), or **ILLEGAL** (violation).
  * Generates an annotated image showing the compliance status and heights.
* **`zoning_db.py` (The Database Interface):** 
  * Manages an SQLite database (`zoning_regulations.db`) containing municipal zoning records (e.g., Residential Zone R-1, Commercial Zone C-1).
  * Includes functions to find the applicable zoning laws for a specific GPS coordinate using Haversine distance calculations.
* **`app.py` (The Web Interface):** 
  * A Flask-based web application that exposes the backend functionality.
  * Features an `/api/analyze` endpoint that accepts image uploads (and optional GPS coordinates), runs the inference pipeline, performs the compliance check, and returns JSON results alongside a base64 encoded annotated image.

## 3. Technology Stack & Requirements
* **Language:** Python 3.10+
* **Deep Learning & Vision:** PyTorch (>=2.1.0), TorchVision, Transformers (Hugging Face), OpenCV (`opencv-python`), Pillow, `pycocotools`.
* **Numerical & Data Science:** NumPy, Pandas, Matplotlib, SciPy.
* **Web Framework:** Flask, Werkzeug.
* **Database:** SQLite3 (Standard Library).

## 4. Current Implementation Status & Progress
The project appears to be a fully functional, highly mature prototype (developed as a Senior Capstone Project).

* **Machine Learning Pipeline:** The inference logic is sophisticated, incorporating multi-model ensembles, depth-guided boundary refinements, and inter-building reference transfers. 
* **Integration:** The backend correctly bridges the ML inference script (`inference_pipeline.py`) with a local rules database (`zoning_db.py`) to produce actionable compliance outputs (`compliance_engine.py`).
* **Web Application:** The Flask API is fully implemented and ready to receive inference requests.
* **Model Training:** Training scripts (`train_2000.py`, `train_july.py`) and various dataset JSON files/Jupyter notebooks are present in the repository, indicating extensive fine-tuning and evaluation phases have been completed.
* **Evaluation:** The `README.md` notes a reference calibration rate of 71% across 270 test images, with 130 verified high-accuracy showcase samples generated.

## 5. Potential Future Enhancements
* Refinement of the front-end interface (HTML/JS) interacting with `app.py`.
* Expansion of the `zoning_db.py` to include real-world, dynamic municipal API integrations rather than a static SQLite database.
* Performance optimizations for the ML inference pipeline (e.g., TensorRT) to reduce response times on the web endpoints.
