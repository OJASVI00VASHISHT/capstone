#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "FS.h"
#include "SD_MMC.h"
#include "credentials.h"   // WiFi SSID, password, SERVER_URL

/**
 * ESP32-CAM Firmware — UrbanVision AI Hardware Feed
 *
 * What this does:
 *   Every 12 seconds (6s × 2 loop iterations), captures a JPEG frame
 *   from the ESP32-CAM and POSTs raw binary to the Flask server at
 *   /api/hardware-upload. The server runs ML analysis in a background
 *   thread and the dashboard auto-updates via polling.
 *
 * Setup:
 *   1. Edit credentials.h with your WiFi name, password, and ngrok URL.
 *   2. Flash this sketch to your AI-Thinker ESP32-CAM.
 *   3. Open the dashboard → click "Live Hardware Feed" tab.
 *   4. Point camera at a building — results appear within ~15 seconds.
 *
 * Camera Pins — AI Thinker ESP32-CAM
 */
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

int captureCounter = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[ESP32] Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[ESP32] Connected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[ESP32] WiFi FAILED. Check credentials.h");
  }
  WiFi.setSleep(false);

  // Camera configuration — SVGA for reasonable quality
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  // SVGA (800x600) — good balance of quality and upload size
  config.frame_size   = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[ESP32] Camera init failed!");
    return;
  }
  Serial.println("[ESP32] Camera ready.");

  // SD Card (optional — fallback storage)
  if (!SD_MMC.begin("/sdcard", true)) {
    Serial.println("[ESP32] SD Card not mounted (optional, continuing).");
  } else {
    Serial.println("[ESP32] SD Card ready.");
  }

  Serial.println("[ESP32] Server URL: " + SERVER_URL);
  Serial.println("[ESP32] Starting capture loop (upload every 12s).");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ESP32] Frame capture failed — retrying.");
    delay(2000);
    return;
  }

  captureCounter++;

  // Upload every capture (alternate with SD save to avoid conflicts)
  if (captureCounter % 2 == 0) {
    uploadToServer(fb);
  } else {
    saveToSD(fb);
  }

  esp_camera_fb_return(fb);
  delay(6000);  // 6s between captures → uploads every 12s
}

// ─── Upload to UrbanVision AI dashboard ────────────────────────────────────
void uploadToServer(camera_fb_t* fb) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ESP32] WiFi lost — skipping upload.");
    return;
  }

  Serial.printf("[ESP32] Uploading frame #%d (%d bytes) to %s\n",
    captureCounter, fb->len, SERVER_URL.c_str());

  WiFiClient client;
  HTTPClient http;
  http.begin(client, SERVER_URL);
  http.setTimeout(15000);   // 15s — server responds immediately but give headroom
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("ngrok-skip-browser-warning", "true");

  int code = http.POST(fb->buf, fb->len);

  if (code == 200) {
    Serial.println("[ESP32] Upload SUCCESS — dashboard will update in ~15s.");
  } else {
    Serial.printf("[ESP32] Upload FAILED — HTTP %d: %s\n", code, http.errorToString(code).c_str());
  }
  http.end();
}

// ─── Save to SD card (local backup on odd captures) ────────────────────────
void saveToSD(camera_fb_t* fb) {
  String path = "/img_" + String(millis()) + ".jpg";
  File file = SD_MMC.open(path.c_str(), FILE_WRITE);
  if (!file) {
    Serial.println("[ESP32] SD write failed.");
    return;
  }
  file.write(fb->buf, fb->len);
  file.flush();
  file.close();
  Serial.println("[ESP32] Saved to SD: " + path);
}