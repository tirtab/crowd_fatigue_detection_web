from flask import Flask, jsonify
from flask_mqtt import Mqtt
from crowd_detector import YOLOv11CrowdDetector
from fatigue_detector import YOLOv11FatigueDetector
import cv2
import logging
import json
from datetime import datetime
import numpy as np
import base64
import time
import threading
import gc

app = Flask(__name__)

# MQTT Configuration
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config["MQTT_KEEP_ALIVE"] = 60
app.config["MQTT_TLS_ENABLED"] = False
# app.config['MQTT_USERNAME'] = ''
# app.config['MQTT_PASSWORD'] = ''
app.config['MQTT_REFRESH_TIME'] = 1.0

# Initialize MQTT
mqtt = Mqtt(app)

# Subscription Topics
CROWD_FRAME_TOPIC = 'mqtt-crowd-frame'
FATIGUE_FRAME_TOPIC = 'mqtt-fatigue-frame'

# Publication Topics
CROWD_RESULT_TOPIC = 'mqtt-crowd-result'
FATIGUE_RESULT_TOPIC = 'mqtt-fatigue-result'

# crowd_detector = YOLOv11CrowdDetector()

# Variabel untuk menyimpan objek deteksi dan frame
num_people = []
status = []
frame_base64 = None


# Load YOLO model
# try:
#     crowd_detector = YOLOv11CrowdDetector()
#     fatigue_detector = YOLOv11FatigueDetector()
# except Exception as e:
#     logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)
#     crowd_detector = None
#     fatigue_detector = None


# Base64 to image
# def process_frame(frame_data):
#     try:
#         if ',' in frame_data:
#             frame_data = frame_data.split(',')[1]
#         frame_bytes = base64.b64decode(frame_data)
#         print(frame_bytes)
#         frame_pil = Image.open(BytesIO(frame_bytes))
#         print(frame_pil)
#         frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
#         print(frame)
#         return frame
#     except Exception as e:
#         logging.error(f"Error processing frame: {e}")
#         return None

# konversi objek numpy.ndarray menjadi list
def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


# Fungsi untuk Menghasilkan Streaming Crowd Analysis
# def generate_crowd_frames(frame):
#     while True:
#         try:
#             frame, detection_data = crowd_detector.detect_and_annotate(frame)
#             num_people = len(detection_data)
#             print(num_people)
#
#             _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
#             frame_base64 = base64.b64encode(buffer).decode("utf-8")
#
#             # Log base64 frame
#             print(
#                 "Frame base64 encoded:", frame_base64[:10]
#             )  # Hanya tampilkan sebagian pertama base64
#
#             # Publikasikan Hasil ke MQTT
#             mqtt_data = {"status": "success",
#                          "timestamp": str(datetime.now()),
#                          "num_people": num_people,
#                          "frame": frame_base64 }
#
#             mqtt.publish(CROWD_RESULT_TOPIC, json.dumps(mqtt_data).encode("utf-8"))
#             print(f"Data sent to MQTT: {json.dumps(mqtt_data)}")
#
#         except Exception as e:
#             logging.error(f"Error dalam memproses frame crowd: {e}")


# Fungsi untuk Menghasilkan Streaming Fatigue Analysis
# def generate_fatigue_frames(frame):
#     try:
#         frame, detected_classes = fatigue_detector.detect_and_annotate(frame)
#         fatigue_status = fatigue_detector.get_fatigue_category(detected_classes)
#
#         # Publikasikan Hasil ke MQTT
#         mqtt_data = {"status": fatigue_status, "timestamp": str(datetime.now())}
#         return mqtt_data
#     except Exception as e:
#         logging.error(f"Error dalam memproses frame fatigue: {e}")


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('Connected to MQTT Broker')

    # Subscribe to topics
    mqtt.subscribe(CROWD_FRAME_TOPIC)
    mqtt.subscribe(FATIGUE_FRAME_TOPIC)

    print(f'Subscribed to {CROWD_FRAME_TOPIC} and {FATIGUE_FRAME_TOPIC}')


@mqtt.on_message()
def handle_mqtt_message(clientt, userdata, message):
    global num_people, status, frame_base64
    topic = message.topic
    try:
        if topic == CROWD_FRAME_TOPIC:
            # string to image
            payload = json.loads(message.payload.decode('utf-8'))
            num_people = payload["num_people"]
            frame_base64 = payload
            {"frame"}
            print(frame_base64)

        # elif topic == FATIGUE_FRAME_TOPIC:
        #     # print(latest_fatigue_frame)
        #     # process fatigue frame and publish result
        #     fatigue_result = process_fatigue_frame(data)
        #     # print(fatigue_result)
        #     mqtt.publish(FATIGUE_RESULT_TOPIC, json.dumps(fatigue_result))

    except json.JSONDecodeError:
        print(f'Error decoding JSON from topic {topic}')
    except Exception as e:
        print(f'Error processing message from {topic}: {e}')


# Endpoint untuk mendapatkan deteksi objek
@app.route("/api/crowd", methods=["GET"])
def get_detections():
    return jsonify({"num_people": num_people, "frame": frame_base64})


# Fungsi untuk memulai loop MQTT
def mqtt_loop():
    mqtt.client.loop_forever()


if __name__ == '__app__':
    # Mulai MQTT loop di thread terpisah
    mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
    mqtt_thread.start()

    # Jalankan deteksi objek di thread terpisah
    detect_thread = threading.Thread(target=generate_crowd_frames, daemon=True)
    detect_thread.start()
    app.run(debug=True, use_reloader=False, port=5000)
