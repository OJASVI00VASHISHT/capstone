from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

SAVE_FOLDER = "images"
os.makedirs(SAVE_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload():

    if not request.data:
        return "No image data", 400

    filename = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f.jpg"
    )

    path = os.path.join(SAVE_FOLDER, filename)

    with open(path, "wb") as f:
        f.write(request.data)

    print("Saved:", filename)
    print("Size:", len(request.data), "bytes")

    return "IMAGE_RECEIVED", 200


@app.route("/")
def home():
    return "Server Running"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )