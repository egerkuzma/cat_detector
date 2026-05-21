"""Cat detector for Raspberry Pi.

Polls an IP camera, runs a TFLite COCO model, turns on LEDs when it's dark
or when a cat shows up at night, and posts the photo to Telegram.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import tflite_runtime.interpreter as tflite
from dotenv import load_dotenv

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    GPIO = None  # allows running off-Pi for syntax checks

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cat_detector")

# --- Config -----------------------------------------------------------------
CAMERA_URL = os.environ["CAMERA_URL"]
CAMERA_USER = os.environ.get("CAMERA_USER")
CAMERA_PASSWORD = os.environ.get("CAMERA_PASSWORD")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SAVE_DIR = Path(os.environ.get("SAVE_DIR", "/home/pi/cam"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/home/pi/detect.tflite"))
LABELS_PATH = Path(os.environ.get("LABELS_PATH", "/home/pi/coco_labels.txt"))

LED_PINS = [int(p) for p in os.environ.get("LED_PINS", "18,23,24").split(",")]
PWM_DUTY_CYCLE = int(os.environ.get("PWM_DUTY_CYCLE", "50"))
LIGHT_ON_DURATION = int(os.environ.get("LIGHT_ON_DURATION", "60"))
LIGHT_ON_CAT_HERE = int(os.environ.get("LIGHT_ON_CAT_HERE", "60"))
BRIGHTNESS_THRESHOLD = int(os.environ.get("BRIGHTNESS_THRESHOLD", "30"))
NOTIFICATION_INTERVAL = int(os.environ.get("NOTIFICATION_INTERVAL", "30"))
NIGHT_START_HOUR = int(os.environ.get("NIGHT_START_HOUR", "22"))
NIGHT_END_HOUR = int(os.environ.get("NIGHT_END_HOUR", "6"))
CAT_CLASS_ID = int(os.environ.get("CAT_CLASS_ID", "17"))  # COCO 'cat'
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.5"))
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL_S", "1"))
HTTP_TIMEOUT_S = float(os.environ.get("HTTP_TIMEOUT_S", "5"))


# --- Hardware ---------------------------------------------------------------
def setup_gpio():
    if GPIO is None:
        log.warning("RPi.GPIO not available — LED control disabled")
        return []
    GPIO.setmode(GPIO.BCM)
    pwms = []
    for pin in LED_PINS:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 1000)
        pwm.start(0)
        pwms.append(pwm)
    return pwms


def set_leds(pwms, duty: int):
    for pwm in pwms:
        pwm.ChangeDutyCycle(duty)


# --- Vision -----------------------------------------------------------------
def capture_frame() -> np.ndarray | None:
    """Fetch one JPEG from the camera. Returns a BGR array or None on failure."""
    auth = (CAMERA_USER, CAMERA_PASSWORD) if CAMERA_USER else None
    try:
        r = requests.get(CAMERA_URL, auth=auth, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("camera fetch failed: %s", e)
        return None
    arr = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_cat(interpreter, frame: np.ndarray) -> bool:
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    shape = in_det[0]["shape"]
    resized = cv2.resize(frame, (shape[1], shape[2]))
    interpreter.set_tensor(in_det[0]["index"], np.expand_dims(resized, 0).astype(np.uint8))
    interpreter.invoke()
    classes = interpreter.get_tensor(out_det[1]["index"])[0]
    scores = interpreter.get_tensor(out_det[2]["index"])[0]
    return any(
        s > DETECTION_THRESHOLD and int(c) == CAT_CLASS_ID
        for s, c in zip(scores, classes)
    )


def mean_brightness(frame: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def is_night() -> bool:
    h = datetime.now().hour
    return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR


# --- Telegram ---------------------------------------------------------------
def send_telegram_photo(path: Path, caption: str = "") -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with path.open("rb") as f:
            requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": f},
                timeout=HTTP_TIMEOUT_S * 2,
            ).raise_for_status()
    except requests.RequestException as e:
        log.warning("telegram send failed: %s", e)


# --- Main loop --------------------------------------------------------------
def main() -> int:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open() as f:
        _ = {i: line.strip() for i, line in enumerate(f.readlines())}

    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    pwms = setup_gpio()

    light_on = False
    light_on_time = 0.0
    last_sent_time = 0.0

    def shutdown(*_):
        log.info("shutting down")
        if pwms:
            set_leds(pwms, 0)
            GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("waiting for cat")
    try:
        while True:
            frame = capture_frame()
            if frame is None:
                time.sleep(POLL_INTERVAL_S)
                continue

            brightness = mean_brightness(frame)
            found = detect_cat(interpreter, frame)
            now = time.time()

            if not light_on and ((found and is_night()) or brightness < BRIGHTNESS_THRESHOLD):
                set_leds(pwms, PWM_DUTY_CYCLE)
                light_on, light_on_time = True, now
                log.info("LEDs on (cat=%s, brightness=%.1f)", found, brightness)

            if light_on:
                window = LIGHT_ON_CAT_HERE if (found and is_night()) else LIGHT_ON_DURATION
                if now - light_on_time > window:
                    set_leds(pwms, 0)
                    light_on = False

            if found and now - last_sent_time > NOTIFICATION_INTERVAL:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                photo = SAVE_DIR / f"cat_{ts}.jpg"
                cv2.imwrite(str(photo), frame)
                send_telegram_photo(photo, caption="Cat sighted")
                last_sent_time = now
                log.info("cat photo sent: %s", photo.name)

            time.sleep(POLL_INTERVAL_S)
    finally:
        if pwms:
            set_leds(pwms, 0)
            GPIO.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
