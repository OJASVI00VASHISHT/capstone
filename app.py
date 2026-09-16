"""
Flask Web Application for Building Height Estimation & Legal Compliance
"""
import os
import sys
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(__file__))

from inference_pipeline import BuildingHeightEstimator
from gps_extractor import extract_gps_from_filename_or_image
from zoning_db import get_all_zones, init_db
from compliance_engine import evaluate_and_annotate_compliance

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

init_db()

print("[Web App] Loading ML Models into memory...")
M1_PATH = os.path.join(os.path.dirname(__file__), "mask_rcnn_model_2000_fixed.pth")
M2_PATH = os.path.join(os.path.dirname(__file__), "mask_rcnn_model_july_400.pth")
estimator = BuildingHeightEstimator(M1_PATH, M2_PATH, fov_deg=95.0)
print("[Web App] Ready for inference requests.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/zones", methods=["GET"])
def list_zones():
    zones = get_all_zones()
    return jsonify({"status": "success", "zones": zones})

@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400
        
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)
    
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
