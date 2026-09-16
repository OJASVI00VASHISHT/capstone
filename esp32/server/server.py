from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Create uploads folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():

    if 'imageFile' not in request.files:
        return "No image found", 400

    file = request.files['imageFile']

    # Timestamp filename
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")

    save_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(save_path)

    print(f"Saved: {save_path}")

    return "Image received successfully", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)