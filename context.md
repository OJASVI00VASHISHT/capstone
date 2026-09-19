# Project Context: Building Height Estimation & Illegal Construction Detection

## 1. Project Overview
This project is an end-to-end computer vision system and web application designed to accurately estimate the metric height (in meters) and floor count of residential and commercial buildings from monocular Google Street View panoramas, standard camera photos, and live hardware feeds (ESP32-CAM). 

The primary goal is to **detect illegal constructions** by comparing the estimated building heights against municipal zoning regulations.

### Key Challenges Addressed
1. **Monocular Depth Scale Ambiguity:** Single 2D images lack absolute metric scaling. 
2. **Foreground Occlusions:** Compound walls, gates, fences, and vehicles can contaminate depth sampling.
3. **Bounding Box Truncation:** Standard instance segmentation often clips roofs or truncates structures.
4. **Hardware Constraints:** Handling image inference from low-powered microcontrollers (ESP32) without causing connection timeouts.

## 2. Technical Solution & Architecture
The system overcomes these challenges without relying on expensive LiDAR or stereo sensors by using a **Reference-Object Depth Ratio Calibration Pipeline**. It leverages the known metric heights of common objects (like cars, pedestrians) to calibrate the depth scale using physical inverse-Z distance projections.

### Core Modules
* **`pipeline/inference_pipeline.py` (The ML Core):** 
  * Loads an ensemble of two fine-tuned Mask R-CNN ResNet-50 FPN models (for building detection) and a standard COCO model (for reference object detection).
  * Uses Hugging Face's `Depth-Anything-V2-Small-hf` for relative depth estimation.
  * Uses a robust roofline and base algorithm to extend truncated bounding boxes.
  * Identifies reference objects (prioritizing ground anchors like vehicles), converts normalized depth values to physical distances (inv_z projection), and applies a mathematical formulation utilizing depth ratios to derive the metric height of the target building.
  * Implements fallback geometric projections (pinhole camera model) if reference objects are absent.
* **`app/compliance_engine.py` (The Legal Evaluator):** 
  * Computes approximate GPS coordinates for the detected buildings based on the camera's location, distance, and azimuth.
  * Cross-references the estimated building heights with municipal zoning limits (fetched from the database).
  * Flags buildings as **LEGAL**, **WARNING** (near limit), or **ILLEGAL** (violation).
  * Generates an annotated image showing the compliance status and heights.
* **`app/zoning_db.py` (The Database Interface):** 
  * Manages an SQLite database (`zoning_regulations.db`) containing municipal zoning records (e.g., Residential Zone R-1, Commercial Zone C-1).
  * Includes functions to find the applicable zoning laws for a specific GPS coordinate using Haversine distance calculations.
* **`app/app.py` (The Web Interface):** 
  * A Flask-based web application that exposes the backend functionality.
  * Features an `/api/analyze` endpoint for manual image uploads.
  * Features an `/api/hardware-upload` endpoint that receives binary JPEG uploads from an ESP32-CAM and processes them in a background thread to prevent hardware timeouts.
  * Provides hardware configuration endpoints (`/api/hardware-config`, `/api/credentials-download`) for generating ready-to-flash `credentials.h` files.
* **Hardware Integration (ESP32-CAM):**
  * `hardware/firmware/espcode.ino`: Captures frames every 12 seconds and POSTs them to the Flask server.
  * Connects securely using credentials generated from the dashboard, removing the need for hardcoded passwords.

## 3. Technology Stack & Requirements
* **Language:** Python 3.10+ (Backend), C++ (Hardware)
* **Deep Learning & Vision:** PyTorch (>=2.1.0), TorchVision, Transformers (Hugging Face), OpenCV (`opencv-python`), Pillow.
* **Numerical & Data Science:** NumPy, math.
* **Web Framework:** Flask, Werkzeug, TailwindCSS, Leaflet.js.
* **Hardware:** AI-Thinker ESP32-CAM.
* **Database:** SQLite3 (Standard Library).

## 4. Current Implementation Status & Progress
The project is a **fully functional, production-ready prototype** ready for its capstone showcase.

### Recent Milestones Achieved:
1. **Directory Refactoring:** Codebase was restructured into a clean, professional hierarchy (`app/`, `hardware/`, `pipeline/`, etc.).
2. **Dashboard UI Rewrite:** A modern, tabbed interface was created (`index.html`) featuring manual upload, sample image quick-launch, and a live hardware feed monitor with automatic polling.
3. **Hardware Configuration:** A modal was added to configure ESP32 settings (WiFi, Server URL, Default GPS) directly from the browser, which generates a `credentials.h` file for flashing. Passwords are saved in a git-ignored `hardware_config.json`.
4. **Thread-Safe Hardware Inference:** The server was upgraded to handle ESP32 POST requests immediately and run the heavy ML pipeline in a background daemon thread, solving a critical 10-second timeout bug.
5. **Depth Math Bug Fix:** Fixed a critical mathematical flaw in `inference_pipeline.py` where normalized depth map values were used directly for scaling ratios. The pipeline now correctly projects normalized depth to physical Z-distances before calculating building scale, preventing height inflation from extreme foreground objects.
6. **Demo Preparation:** A comprehensive `DEMO_GUIDE.md` was created, detailing exact steps to present both the manual sample analysis and live hardware feed.

## 5. Potential Future Enhancements
* Move database to a cloud provider (e.g., PostgreSQL) for real-world dynamic municipal API integrations.
* Performance optimizations for the ML inference pipeline (e.g., TensorRT or ONNX export) to reduce inference time below the current ~10s CPU limit.
* Expand the ESP32 firmware to support WebSockets for lower-latency streaming.
