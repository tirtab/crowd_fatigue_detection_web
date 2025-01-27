from flask import Flask, jsonify
from flask_mqtt import Mqtt
from crowd_detector import YOLOv11CrowdDetector
from fatigue_detector import YOLOv11FatigueDetector
from PIL import Image
from io import BytesIO
import cv2
import logging
import json
import numpy as np
import base64


from ultralytics import YOLO

app = Flask(__name__)

# try:
    # crowd_detector = YOLOv11CrowdDetector()
    # crowd_detector = YOLOv11FatigueDetector()
    # print(crowd_detector)
    # model = YOLO('yolov8n.pt')
# except Exception as e:
#     logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)

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
        data = json.loads(payload)

        if topic == CROWD_FRAME_TOPIC:
            latest_crowd_frame = data
            # process crowd frame and publish result
            mqtt.publish(CROWD_RESULT_TOPIC, json.dumps({'message': 'test-crowd'}))

        elif topic == FATIGUE_FRAME_TOPIC:
            latest_fatigue_frame = data
            # process fatigue frame and publish result
            mqtt.publish(CROWD_RESULT_TOPIC, json.dumps({'detection_data': 'test-fatigue'}))


    except json.JSONDecodeError:
        print(f'Error decoding JSON from topic {topic}')
    except Exception as e:
        print(f'Error processing message from {topic}: {e}')



if __name__ == '__app__':
    app.run(debug=True)
