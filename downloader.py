from fastapi import FastAPI
import requests

app = FastAPI()

ESP32_IP = "172.20.10.3"

@app.get("/led/on")
def led_on():
    r = requests.get(f"http://{ESP32_IP}/led?state=1")
    return {"esp_response": r.text}

@app.get("/led/off")
def led_off():
    r = requests.get(f"http://{ESP32_IP}/led?state=0")
    return {"esp_response": r.text}