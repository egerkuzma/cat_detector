# 🐱 Детектор кота с уведомлениями

Скрипт для Raspberry Pi, который отслеживает появление кота через IP-камеру, включает освещение при низкой освещенности и отправляет уведомления в Telegram.

## 📋 Возможности

- Обнаружение кота с помощью TensorFlow Lite
- Автоматическое включение LED-освещения в темное время
- Сохранение фотографий кота
- Отправка уведомлений в Telegram
- Управление мощностью LED через PWM

## 🛠 Требования

### Оборудование
- Raspberry Pi (любая модель)
- IP-камера
- 3 белых светодиода
- Резисторы для светодиодов
- GPIO-подключения

### Программные зависимости

bash
Установка необходимых пакетов
sudo apt-get update
sudo apt-get install -y python3-pip wget curl
sudo apt-get install -y libatlas-base-dev # для numpy
sudo apt-get install -y libjpeg-dev # для Pillow/opencv
Установка Python-пакетов
pip3 install opencv-python-headless
pip3 install numpy
pip3 install tflite-runtime
pip3 install RPi.GPIO


wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite /home/pi/
mv labelmap.txt /home/pi/coco_labels.txt


## ⚙️ Настройка

1. Скопируйте файлы:
   - `detect.tflite` (модель TensorFlow Lite)
   - `coco_labels.txt` (файл с метками классов COCO)

2. Настройте параметры в скрипте:
   - `CAMERA_URL`: URL вашей IP-камеры
   - `BOT_TOKEN`: токен вашего Telegram-бота
   - `CHAT_ID`: ID чата для уведомлений
   - `LED_PINS`: GPIO-пины для светодиодов

3. Создайте необходимые директории:
   ```bash
   mkdir -p /home/pi/cam
   ```

## 🚀 Запуск

bash
python3 cat_detector4.py


## 📝 Примечания

- Скрипт использует модель COCO, где кот определяется как класс 17
- Уведомления отправляются не чаще чем раз в 10 секунд
- Светодиоды работают на 50% мощности для защиты GPIO
- Освещение автоматически выключается через 60 секунд

## 🔧 Устранение неполадок

- Убедитесь, что IP-камера доступна по сети
- Проверьте правильность подключения светодиодов
- Убедитесь, что все пути к файлам корректны
- Проверьте права доступа к директориям для сохранения фото

## 📜 Лицензия

MIT

