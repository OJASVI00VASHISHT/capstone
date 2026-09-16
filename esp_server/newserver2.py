from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():

    if 'imageFile' not in request.files:
        return "No image found", 400

    file = request.files['imageFile']

    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    esp_timestamp = request.form.get("timestamp")

    server_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{server_time}.jpg"

    save_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(save_path)

    print("IMAGE SAVED")
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    print("ESP Timestamp:", esp_timestamp)

    return "Upload successful", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)