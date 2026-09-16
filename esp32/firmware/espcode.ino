#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "FS.h"
#include "SD_MMC.h"

// WiFi
const char* ssid = "Ojasvi";
const char* password = "12345678";

// ✅ FINAL URL
// Since ngrok's HTTPS requires too much memory, you must use a standard HTTP link.
// 1. Open a new Command Prompt and run:  npx localtunnel --port 5000
// 2. Copy the "your url is: http://..." link and paste it below. Make sure it starts with HTTP, not HTTPS!
String serverUrl = "http://YOUR_LOCALTUNNEL_LINK/upload/";

// Camera Pins (AI Thinker)
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

int counter = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // WiFi connect
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");

  WiFi.setSleep(false);

  // Camera config
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
//resolution change here plz
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 15;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  // SD Card init (1-bit mode)
  if (!SD_MMC.begin("/sdcard", true)) {
    Serial.println("SD Card Mount Failed");
  } else {
    Serial.println("SD Card initialized");
  }
}

void loop() {
  camera_fb_t * fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // 🔥 Alternate operations to avoid conflict
  if (counter % 2 == 0) {
    uploadImage(fb);
  } else {
    saveToSD(fb);
  }

  counter++;

  esp_camera_fb_return(fb);

  delay(6000);
}

// ================= SAVE TO SD =================
void saveToSD(camera_fb_t * fb) {
  delay(100);

  String path = "/img_" + String(millis()) + ".jpg";

  File file = SD_MMC.open(path.c_str(), FILE_WRITE);
  if (!file) {
    Serial.println("SD Write Failed");
    return;
  }

  file.write(fb->buf, fb->len);
  file.flush();
  file.close();

  Serial.println("Saved to SD: " + path);
}

// ================= UPLOAD =================
void uploadImage(camera_fb_t * fb) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected");
    return;
  }

  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  Serial.print("Image size (bytes): ");
  Serial.println(fb->len);

  WiFiClient client;
  HTTPClient http;
  http.begin(client, serverUrl);
  http.setTimeout(10000);
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("ngrok-skip-browser-warning", "true");

  int code = http.POST(fb->buf, fb->len);

  Serial.print("HTTP Code: ");
  Serial.println(code);
  Serial.print("Error string: ");
  Serial.println(http.errorToString(code));
  Serial.print("Response: ");
  Serial.println(http.getString());

  if (code == 200) Serial.println("Upload SUCCESS");
  else Serial.println("Upload FAILED");

  http.end();
}