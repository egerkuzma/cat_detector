# cat_detector

Raspberry Pi script that watches an IP camera for a cat, lights up an LED strip when it's dark, and sends a photo to Telegram.

![python](https://img.shields.io/badge/python-3.9+-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## What it does

A small loop runs on a Raspberry Pi: every second it grabs a JPEG from an IP camera over HTTP, feeds it to a quantised TensorFlow Lite COCO SSD MobileNet, and looks for class 17 (cat). If it sees one and it's after 22:00 local time, it turns on three PWM-driven LEDs for 60 seconds and posts the photo to a Telegram chat. The same LEDs also turn on whenever the scene is too dark, regardless of cats.

This is a hobby project for one specific kitchen camera, not a generic surveillance tool.

## Hardware

| Part | Notes | Approx. cost |
| --- | --- | --- |
| Raspberry Pi (any 3B+ or newer) | Tested on Pi 4 | $35 |
| IP camera with HTTP JPEG snapshot | Anything that serves `/IMAGE.JPG` | $20+ |
| 3 × white LED | Diffused 5mm or strip | $1 |
| 3 × current-limiting resistor (~220 Ω) | One per LED | <$1 |

LEDs connect to GPIO 18, 23, 24 by default. Each LED has its own GND.

## Quickstart

```bash
git clone https://github.com/egerkuzma/cat_detector.git
cd cat_detector
pip install -r requirements.txt
# Download the COCO SSD MobileNet model
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite ~/ && mv labelmap.txt ~/coco_labels.txt
cp .env.example .env  # then edit
python cat_detector.py
```

## Configuration

All settings live in environment variables (or a local `.env` file). See [.env.example](.env.example).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAMERA_URL` | _required_ | HTTP URL that returns a JPEG |
| `CAMERA_USER` / `CAMERA_PASSWORD` | _empty_ | HTTP basic auth for the camera |
| `TELEGRAM_BOT_TOKEN` | _required_ | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | _required_ | Chat ID for notifications |
| `LED_PINS` | `18,23,24` | BCM pin numbers |
| `PWM_DUTY_CYCLE` | `50` | Percent power for LEDs |
| `BRIGHTNESS_THRESHOLD` | `30` | Mean grayscale below which "dark" |
| `NIGHT_START_HOUR` / `NIGHT_END_HOUR` | `22` / `6` | "Night" window in local hours |
| `NOTIFICATION_INTERVAL` | `30` | Min seconds between Telegram posts |
| `LIGHT_ON_DURATION` | `60` | Seconds LEDs stay on if just dark |
| `LIGHT_ON_CAT_HERE` | `60` | Seconds LEDs stay on for cat at night |

## How it works

```
   IP cam ──HTTP──▶ requests.get
                       │
                       ▼
                  cv2.imdecode ──▶ TFLite (COCO SSD MobileNet, INT8)
                                       │
                                       ├──▶ class 17 + score > 0.5 ──▶ Telegram POST
                                       │
                                       └──▶ LED PWM (RPi.GPIO)
```

The model is the standard quantised COCO SSD MobileNet v1 from Google's TFLite samples — COCO class 17 is `cat`. No retraining.

## Limitations

- Polls once per second over HTTP. Wasteful if your camera supports MJPEG or RTSP.
- COCO MobileNet is small and quantised. Expect false positives on other small mammals and missed cats in poor lighting.
- Lighting logic is local-time-based, not solar — adjust `NIGHT_START_HOUR` for your latitude/season.
- No persistence: a restart resets the cooldown timer.
- Originally written for a single specific IP camera that exposes `/IMAGE.JPG?cidx=...` over plain HTTP. URLs vary by vendor.

## Credits

- [TensorFlow Lite COCO SSD MobileNet](https://www.tensorflow.org/lite/examples/object_detection/overview)
- [RPi.GPIO](https://pypi.org/project/RPi.GPIO/) for PWM

## License

MIT — see [LICENSE](LICENSE).
