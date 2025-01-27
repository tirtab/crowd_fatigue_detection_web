from flask import Flask, jsonify
from flask_mqtt import Mqtt
from fatigue_detector import YOLOv11FatigueDetector
from crowd_detector import YOLOv11CrowdDetector
from PIL import Image
from io import BytesIO
import cv2
import logging
import json
from datetime import datetime
import numpy as np
import base64
import time
import threading
import gc

from ultralytics import YOLO

app = Flask(__name__)

# initialize model
try:
    crowd_detector = YOLOv11CrowdDetector()
    # fatigue_detector = YOLOv11FatigueDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)

# MQTT Configuration
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
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

# Global variables to store latest received messages
latest_crowd_frame = None
latest_fatigue_frame = None


# Base64 to image
# Fungsi untuk Memproses Frame dari Data Base64
def process_frame(frame_data):
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_pil = Image.open(BytesIO(frame_bytes))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None


# konversi objek numpy.ndarray menjadi list
def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Konversi ke tipe float native Python
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Konversi ke tipe int native Python
    raise TypeError(f"Type {type(obj)} not serializable")


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('Connected to MQTT Broker')

    # Subscribe to topics
    mqtt.subscribe(CROWD_FRAME_TOPIC)
    mqtt.subscribe(FATIGUE_FRAME_TOPIC)

    print(f'Subscribed to {CROWD_FRAME_TOPIC} and {FATIGUE_FRAME_TOPIC}')


@mqtt.on_message()
def handle_mqtt_message(clientt, userdata, message):
    global latest_crowd_frame, latest_fatigue_frame
    topic = message.topic
    payload = message.payload.decode('utf-8')

    try:
        # parse the payload
        data = json.loads(payload.replace("'", '"'))
        frame_data = data['frame']
        id = data['id']

        if topic == CROWD_FRAME_TOPIC:
            latest_crowd_frame = frame_data
            # print(latest_crowd_frame)

            # proccess frame
            frame = process_frame(latest_crowd_frame)

            frame, detection_data = crowd_detector.detect_and_annotate(frame)
            num_people = len(detection_data)

            # process crowd frame and publish result
            mqtt.publish(f'{CROWD_RESULT_TOPIC}-{id}', json.dumps({
                'detection_data': detection_data,
                'num_people': num_people
            }))

        # elif topic == FATIGUE_FRAME_TOPIC:
        #     latest_fatigue_frame = data
        #     # print(latest_fatigue_frame)
        #
        #     # process fatigue frame and
        #     frame = process_frame(data)
        #
        #     frame, detection_results = fatigue_detector.detect_and_annotate(frame)
        #     fatigue_status = fatigue_detector.get_fatigue_category(detection_results)
        #
        #     # publish result
        #     mqtt.publish(FATIGUE_RESULT_TOPIC, json.dumps({
        #         'detection_result': detection_results,
        #         'fatigue_status': fatigue_status
        #     }, default=custom_serializer))

    except json.JSONDecodeError:
        print(f'Error decoding JSON from topic {topic}')
    except Exception as e:
        print(f'Error processing message from {topic}: {e}')


if __name__ == '__app__':
    app.run(debug=True)
