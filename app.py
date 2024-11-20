# analytics/app.py
from flask import Flask
import paho.mqtt.client as mqtt
import json

from datetime import datetime
import logging

import base64
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

# Model
from yolov11_detector import YOLOv11CrowdDetector

app = Flask(__name__)

# Load YOLO model
try:
    detector = YOLOv11CrowdDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)
    detector = None


# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)


def process_frame(frame_data):
    try:
        frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_pil = Image.open(BytesIO(frame_bytes))
        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        frame, detection_data = detector.detect_and_annotate(frame_cv2)
        num_people = len(detection_data)

        print(type(detection_data))
        detection_summary = {
            "num_people": num_people,
            "detections": detection_data  # Tambahkan semua data bounding box dan confidence
        }

        print(detection_summary)

        return {
            "status": "true",
            "data": {
                detection_data
            },
            "timestamp": str(datetime.now()),
        }

    except Exception as e:
        return {
            "status": "error",
            "timestamp": str(datetime.now()),
            "error": str(e)
        }


def on_message(client, userdata, message):
    try:
        # Process frame
        frame_data = message.payload.decode()
        detections = process_frame(frame_data)

        # Publish results back
        mqtt_client.publish('video-analysis', json.dumps(detections))
    except Exception as e:
        print(f"Error in MQTT message handling: {str(e)}")


# Set up MQTT subscriber
mqtt_client.on_message = on_message
mqtt_client.subscribe('video-frames')
mqtt_client.loop_start()

if __name__ == '__main__':
    app.run(port=5000)