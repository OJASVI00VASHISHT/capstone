"""
Flask Web Application for Building Height Estimation & Legal Compliance
Supports: Manual image upload, ESP32 Live Hardware Feed, Sample image quick-launch
"""
import os
import sys
import uuid
import json
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
from datetime import datetime

# Add repo root to path (for pipeline/ package) and app/ itself (for sibling modules)
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _APP_DIR)

from pipeline.inference_pipeline import BuildingHeightEstimator
from gps_extractor import extract_gps_from_filename_or_image
from zoning_db import get_all_zones, init_db
from compliance_engine import evaluate_and_annotate_compliance

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(_APP_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

init_db()

print("[Web App] Loading ML Models into memory...")
M1_PATH = os.path.join(_REPO_ROOT, "mask_rcnn_model_2000_fixed.pth")
M2_PATH = os.path.join(_REPO_ROOT, "mask_rcnn_model_july_400.pth")
estimator = BuildingHeightEstimator(M1_PATH, M2_PATH, fov_deg=95.0)
print("[Web App] Ready for inference requests.")

# -----------------------------------------------------------------------
# Hardware Config — persisted to hardware_config.json at repo root
# -----------------------------------------------------------------------
_HW_CONFIG_PATH = os.path.join(_REPO_ROOT, "hardware_config.json")
_DEFAULT_HW_CONFIG = {
    "wifi_ssid": "",
    "wifi_password": "",
    "server_url": "",
    "default_lat": 28.6139,
    "default_lon": 77.2090
}

def _load_hw_config():
    if os.path.exists(_HW_CONFIG_PATH):
        try:
            with open(_HW_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg = dict(_DEFAULT_HW_CONFIG)
                cfg.update(saved)
                return cfg
        except Exception:
            pass
    return dict(_DEFAULT_HW_CONFIG)

def _save_hw_config(cfg):
    with open(_HW_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

_hw_config = _load_hw_config()

# -----------------------------------------------------------------------
# Global state for ESP32 Live Hardware Feed (thread-safe)
# -----------------------------------------------------------------------
_hw_lock = threading.Lock()
_latest_hw = {
    "status": "waiting",   # waiting | processing | ready | error
    "timestamp": None,
    "data": None
}


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/zones", methods=["GET"])
def list_zones():
    zones = get_all_zones()
    return jsonify({"status": "success", "zones": zones})


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns a list of filenames available in sample_images/ for quick-launch."""
    samples_dir = os.path.join(_REPO_ROOT, "sample_images")
    files = []
    if os.path.exists(samples_dir):
        for f in sorted(os.listdir(samples_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                files.append(f)
    return jsonify({"files": files})


@app.route("/api/samples/<path:filename>", methods=["GET"])
def serve_sample(filename):
    """Serves a sample image file directly from sample_images/."""
    samples_dir = os.path.join(_REPO_ROOT, "sample_images")
    return send_from_directory(samples_dir, filename)


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Main endpoint: accepts uploaded image + optional GPS, returns compliance JSON."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Validate image is readable
    test_img = cv2.imread(save_path)
    if test_img is None:
        return jsonify({"status": "error", "message": "Uploaded file is not a valid image"}), 400

    custom_lat = request.form.get("latitude")
    custom_lon = request.form.get("longitude")

    if custom_lat and custom_lon and custom_lat.strip() and custom_lon.strip():
        try:
            lat = float(custom_lat)
            lon = float(custom_lon)
            gps_source = "Manual GPS Input"
        except ValueError:
            lat, lon, gps_source = extract_gps_from_filename_or_image(file.filename)
    else:
        lat, lon, gps_source = extract_gps_from_filename_or_image(file.filename)

    try:
        annotated_bgr, buildings = estimator.estimate(save_path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference error: {str(e)}"}), 500

    compliance_data = evaluate_and_annotate_compliance(annotated_bgr, buildings, lat, lon)

    return jsonify({
        "status": "success",
        "filename": filename,
        "gps": {
            "latitude": lat,
            "longitude": lon,
            "source": gps_source
        },
        "compliance": compliance_data
    })


@app.route("/api/hardware-upload", methods=["POST"])
def hardware_upload():
    """
    ESP32 endpoint: receives raw binary JPEG via POST.
    Saves image then runs ML inference asynchronously in a background thread
    so the ESP32 gets an immediate "IMAGE_RECEIVED" and never times out.
    Dashboard polls /api/latest to get the result.
    """
    img_data = request.data
    if not img_data or len(img_data) < 100:
        return "No image data", 400

    # Save raw binary to disk
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], "esp_latest.jpg")
    with open(save_path, "wb") as f:
        f.write(img_data)

    # GPS priority: query params -> saved dashboard config -> Delhi default
    try:
        lat = float(request.args.get("lat", _hw_config["default_lat"]))
        lon = float(request.args.get("lon", _hw_config["default_lon"]))
    except (ValueError, KeyError):
        lat, lon = 28.6139, 77.2090

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Mark as processing immediately
    with _hw_lock:
        _latest_hw["status"] = "processing"
        _latest_hw["timestamp"] = ts
        _latest_hw["data"] = None

    # Run full ML pipeline in background thread
    def _run_analysis(img_path, cam_lat, cam_lon, capture_time):
        try:
            img_check = cv2.imread(img_path)
            if img_check is None:
                raise ValueError("Saved ESP32 image is not readable by OpenCV")

            annotated_bgr, buildings = estimator.estimate(img_path)
            compliance_data = evaluate_and_annotate_compliance(annotated_bgr, buildings, cam_lat, cam_lon)

            with _hw_lock:
                _latest_hw["status"] = "ready"
                _latest_hw["timestamp"] = capture_time
                _latest_hw["data"] = {
                    "gps": {
                        "latitude": cam_lat,
                        "longitude": cam_lon,
                        "source": "ESP32 Hardware"
                    },
                    "compliance": compliance_data
                }
        except Exception as exc:
            with _hw_lock:
                _latest_hw["status"] = "error"
                _latest_hw["timestamp"] = capture_time
                _latest_hw["data"] = {"error": str(exc)}

    t = threading.Thread(
        target=_run_analysis,
        args=(save_path, lat, lon, ts),
        daemon=True
    )
    t.start()

    print(f"[Hardware] Image received ({len(img_data)} bytes) at {ts} — analysis started in background")
    return "IMAGE_RECEIVED", 200


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """
    Dashboard polling endpoint. Returns the latest hardware analysis result.
    Status values: 'waiting' | 'processing' | 'ready' | 'error'
    """
    with _hw_lock:
        result = dict(_latest_hw)
    return jsonify(result)


@app.route("/api/hardware-config", methods=["GET"])
def get_hw_config():
    """Returns saved hardware config (passwords are masked for security)."""
    safe = dict(_hw_config)
    if safe.get("wifi_password"):
        safe["wifi_password"] = "*" * len(safe["wifi_password"])
        safe["wifi_password_set"] = True
    else:
        safe["wifi_password_set"] = False
    return jsonify({"status": "success", "config": safe})


@app.route("/api/hardware-config", methods=["POST"])
def save_hw_config():
    """Saves hardware config. Accepts JSON body."""
    global _hw_config
    body = request.get_json(force=True, silent=True) or {}
    updated = dict(_hw_config)
    if "wifi_ssid" in body:
        updated["wifi_ssid"] = str(body["wifi_ssid"]).strip()
    if "wifi_password" in body and body["wifi_password"] and not all(c == '*' for c in body["wifi_password"]):
        updated["wifi_password"] = str(body["wifi_password"])
    if "server_url" in body:
        updated["server_url"] = str(body["server_url"]).strip().rstrip("/")
    if "default_lat" in body:
        try:
            updated["default_lat"] = float(body["default_lat"])
        except (ValueError, TypeError):
            pass
    if "default_lon" in body:
        try:
            updated["default_lon"] = float(body["default_lon"])
        except (ValueError, TypeError):
            pass
    _hw_config = updated
    _save_hw_config(updated)
    print(f"[Config] Hardware config saved: SSID={updated['wifi_ssid']}, URL={updated['server_url']}, GPS=({updated['default_lat']},{updated['default_lon']})")
    return jsonify({"status": "success", "message": "Configuration saved."})


@app.route("/api/credentials-download", methods=["GET"])
def download_credentials():
    """
    Generates and serves a ready-to-use credentials.h file for the ESP32.
    Pulls WiFi SSID, password, and server URL from saved config.
    """
    cfg = _hw_config
    server_url = cfg.get("server_url", "").rstrip("/")
    if server_url and not server_url.endswith("/api/hardware-upload"):
        server_url = server_url + "/api/hardware-upload"
    elif not server_url:
        server_url = "http://YOUR_SERVER_IP:5000/api/hardware-upload"

    content = f'''/**
 * credentials.h — Generated by UrbanVision AI Dashboard
 * Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 *
 * Flash this file alongside espcode.ino to your ESP32-CAM.
 * Do NOT commit this file to Git (it is in .gitignore).
 */
#ifndef CREDENTIALS_H
#define CREDENTIALS_H

const char* WIFI_SSID     = "{cfg.get('wifi_ssid', '')}";       // WiFi network name
const char* WIFI_PASSWORD = "{cfg.get('wifi_password', '')}";   // WiFi password

// Full URL to the Flask server hardware endpoint
String SERVER_URL = "{server_url}";

#endif
'''
    resp = make_response(content)
    resp.headers["Content-Type"] = "text/plain"
    resp.headers["Content-Disposition"] = "attachment; filename=credentials.h"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
