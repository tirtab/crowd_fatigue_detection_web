import threading
import time
import json
import base64
from flask_mqtt import Mqtt
from flask import Flask
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

app = Flask(__name__)


# MQTT Configuration
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
# app.config['MQTT_USERNAME'] = ''
# app.config['MQTT_PASSWORD'] = ''
app.config['MQTT_REFRESH_TIME'] = 1.0

# Initialize MQTT
mqtt = Mqtt(app)

# Subscription Topics
FRAME_TOPIC = 'mqtt-face-frame'
RESULT_TOPIC = 'mqtt-face-result'


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('Connected to MQTT Broker')

    # Subscribe to topics
    mqtt.subscribe(FRAME_TOPIC)
    mqtt.subscribe(RESULT_TOPIC)

    print(f'Subscribed to {FRAME_TOPIC} and {RESULT_TOPIC}')

@mqtt.on_message()
def handle_mqtt_message(clientt, userdata, message):
    global latest_crowd_frame, latest_fatigue_frame
    topic = message.topic
    payload = message.payload.decode('utf-8')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

if __name__ == '__app__':
    app.run(debug=True, port=5000)
