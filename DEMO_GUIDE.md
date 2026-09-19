# Demo Day Guide — UrbanVision AI

## Before Demo Day (one-time setup)

### 1. Install ngrok
Download from https://ngrok.com/download
Sign up free → get auth token → run: ngrok config add-authtoken YOUR_TOKEN

### 2. Install Python dependencies
```
cd "c:\Users\rauna\Videos\Pulkit Project\Illegal-Construction-Detection"
pip install flask werkzeug
```

---

## On Demo Day — Step by Step

### Terminal 1 — Start the Flask App
```
python app/app.py
```
Wait ~30 seconds for "Ready for inference requests." Keep this running.

### Terminal 2 — Expose via ngrok
```
ngrok http 5000
```
Copy the  https://xxx.ngrok-free.app  URL.

### Open Dashboard
- Your laptop: http://localhost:5000
- Teacher device: https://xxx.ngrok-free.app

---

## Demo Scenario A — Sample Images (No Hardware Needed)
1. Dashboard opens on Manual Upload tab by default
2. Scroll to "Quick Demo — Sample Images" section
3. Click any thumbnail
4. Wait ~15 seconds -> result appears
5. Show teacher: annotated image, LEGAL/ILLEGAL badge, GPS map, building breakdown

Best sample images:
- building_4floor_...ZoneR1_IllegalViolation.png  => ILLEGAL result (4 floors > 3 allowed)
- house_2floor_...ZoneR1_Legal.png               => LEGAL result (clean pass)
- commercial_...ZoneC1_Legal.png                 => Commercial zone (15m limit)

---

## Demo Scenario B — Live ESP32 Hardware Feed

### Before demo: Flash ESP32
1. Open hardware/firmware/credentials.h
2. Set your WiFi name, password, and ngrok URL + /api/hardware-upload
3. Flash espcode.ino + credentials.h via Arduino IDE
4. Serial Monitor (115200 baud) should show "Upload SUCCESS"

### During demo:
1. Click "Live Hardware Feed" tab
2. Point ESP32 at a building/wall/object
3. Every 12s: image uploads -> analysis runs in background (~15s)
4. Hardware Status dot: amber (analyzing) -> green (done)
5. Result auto-appears: annotated image + compliance badge

---

## Architecture (for teacher)

ESP32-CAM -> POST raw JPEG -> /api/hardware-upload
                                    |
                          Background thread: ML pipeline
                          (Depth-Anything-V2 + Mask R-CNN x2)
                                    |
Browser polls /api/latest every 5s -> renders result live

---

## Troubleshooting
- Models slow to load: Normal, wait 60s on first start
- ngrok URL expired: Restart ngrok, update credentials.h, reflash
- Analysis takes >20s: Normal on CPU (no GPU)
- Sample images missing: Check sample_images/ folder has .png/.jpg files
