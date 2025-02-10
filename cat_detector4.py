import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import RPi.GPIO as GPIO
from datetime import datetime

# GPIO-пины для 3 белых светодиодов
LED_PINS = [18, 23, 24]  # Каждый светодиод подключён к разному GND
PWM_DUTY_CYCLE = 50  # 50% мощности, чтобы не перегружать GPIO


LIGHT_ON_CAT_HERE = 60 # 60 секунд, если кот обнаружен, то свет включается на 60 секунд

# Время работы светодиодов (5 минут = 300 секунд)
LIGHT_ON_DURATION = 60 #освещение включается на 60 секунд
light_on_time = 0  # Когда включили свет
light_on = False  # Флаг состояния света

# Настраиваем GPIO
GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)

# Создаём PWM для ограничения мощности
pwm_leds = [GPIO.PWM(pin, 1000) for pin in LED_PINS]
for pwm in pwm_leds:
    pwm.start(0)  # Светодиоды выключены по умолчанию

# Пути к файлам
IMAGE_PATH = "/home/pi/cam/image.jpg"
SAVE_PATH = "/home/pi/cam/"
MODEL_PATH = "/home/pi/detect.tflite"
LABELS_PATH = "/home/pi/coco_labels.txt"

# URL камеры
CAMERA_URL = "http://ip/IMAGE.JPG?cidx=$(date +%s)"
WGET_COMMAND = f'wget --user=admin --password="" -q -O {IMAGE_PATH} "{CAMERA_URL}"'
# Данные для Telegram
BOT_TOKEN = "token"
CHAT_ID = "chat_id"
TELEGRAM_PHOTO_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

# Время последней отправки
last_sent_time = 0
NOTIFICATION_INTERVAL = 30  #каждые 30 сек отправлять кота в телегу, он может просто придти и не поесть

# Загружаем классы (кот = 17 в COCO)
with open(LABELS_PATH, "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Загружаем модель
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("⏳ Ожидание кота...")

def check_brightness(image):
    """ Проверяем среднюю яркость изображения """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print(f"💡 Текущая освещённость: {brightness}")
    return brightness

def turn_on_leds():
    """ Включаем светодиоды (ограниченная яркость) """
    for pwm in pwm_leds:
        pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)  # 50% мощности
    print("💡 Светодиоды ВКЛЮЧЕНЫ")

def turn_off_leds():
    """ Выключаем светодиоды """
    for pwm in pwm_leds:
        pwm.ChangeDutyCycle(0)  # 0% мощности
    print("💡 Светодиоды ВЫКЛЮЧЕНЫ")

def is_night_time():
    """Проверяем, ночное ли сейчас время (22:00-06:00)"""
    current_hour = datetime.now().hour
    return current_hour >= 22 or current_hour < 6

# Основной цикл проверки изображений
while True:
    os.system(WGET_COMMAND)

    if not os.path.exists(IMAGE_PATH):
        print("⚠️ Файл изображения не найден!")
        time.sleep(1)
        continue

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("⚠️ Ошибка чтения изображения!")
        time.sleep(1)
        continue

    # Проверяем освещенность
    brightness = check_brightness(frame)

    # Запускаем нейросеть
    input_shape = input_details[0]['shape']
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Запускаем нейросеть
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Получаем результаты
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Проверяем, есть ли кот (класс 17)
    found_cat = any(scores[0][i] > 0.5 and int(classes[0][i]) == 17 for i in range(len(scores[0])))

    current_time = time.time()

    # Проверяем освещенность и наличие кота
    brightness = check_brightness(frame)
    
    # Новая логика включения света
    if found_cat and is_night_time() and not light_on:
        # Если кот обнаружен ночью - включаем свет сразу
        turn_on_leds()
        light_on = True
        light_on_time = current_time
        print("🌙 Кот обнаружен ночью - включаем подсветку")
    elif brightness < 30 and not light_on:
        # Стандартная логика включения по освещенности
        turn_on_leds()
        light_on = True
        light_on_time = current_time
        print(f"🌑 Низкая освещенность ({brightness:.1f}) - включаем подсветку")

    # Проверяем, нужно ли выключить свет
    if light_on:
        if found_cat and is_night_time():
            # Если кот есть и сейчас ночь - используем LIGHT_ON_CAT_HERE
            if current_time - light_on_time > LIGHT_ON_CAT_HERE:
                turn_off_leds()
                light_on = False
        else:
            # В остальных случаях используем стандартный LIGHT_ON_DURATION
            if current_time - light_on_time > LIGHT_ON_DURATION:
                turn_off_leds()
                light_on = False

    if found_cat and (current_time - last_sent_time > NOTIFICATION_INTERVAL):
        print("🐱 КОТ ОБНАРУЖЕН!")

        # Сохраняем фото
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cat_photo_path = os.path.join(SAVE_PATH, f"shkoda_{timestamp}.jpg")
        cv2.imwrite(cat_photo_path, frame)
        print(f"📸 Фото сохранено: {cat_photo_path}")

        # Отправляем фото в Telegram
        send_photo_cmd = f'curl -s -X POST "{TELEGRAM_PHOTO_URL}" -F chat_id="{CHAT_ID}" -F photo="@{cat_photo_path}" -F caption="🐱 Кот пришел!"'
        os.system(send_photo_cmd)
        last_sent_time = current_time  # Обновляем время последней отправки
        print("📨 Фото и сообщение отправлены в Telegram!")

    time.sleep(1)  # Проверяем раз в секунду
