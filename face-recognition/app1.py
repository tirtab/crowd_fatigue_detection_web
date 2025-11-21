import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from dotenv import load_dotenv
import os
import json
import base64
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from io import BytesIO
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

from flask import Flask
from flask_mqtt import Mqtt

# Flask and MQTT setup
app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0

mqtt = Mqtt(app)



# Topics
FRAME_TOPIC = 'mqtt-face-frame'
RESULT_TOPIC = 'mqtt-face-result'


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker")
    mqtt.subscribe(FRAME_TOPIC)
    print(f"Subscribed to {FRAME_TOPIC}")


# Load .env file
load_dotenv()

# Global variables
stop_threads = False
id_face_mapping = {}
latest_frame = None


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

# D:\magang\comvis-CnFD\crowd_fatigue_detection_web\face-recognition\app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gunakan variabel environment
face_model_path = os.path.join(BASE_DIR, "face_detection", "scrfd", "weights", "scrfd_2.5g_bnkps.onnx")
arc_face_model_path = os.path.join(BASE_DIR, "face_recognition", "arcface", "weights", "arcface_r100.pth")
face_feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "feature")
config_tracking_path = os.path.join(BASE_DIR, "face_tracking", "config", "config_tracking.yaml")


config_tracking = load_config(config_tracking_path)

# Initialize models
detector = SCRFD(model_file=face_model_path)
recognizer = iresnet_inference(
    model_name="r100",
    path=arc_face_model_path,
    device=device
)
images_names, images_embs = read_features(feature_path=face_feature_path)


def process_frame(frame_data):
    try:
        # Check if frame_data is None or empty
        if not frame_data:
            return None

        # Handle base64 string format
        if isinstance(frame_data, str):
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            frame_bytes = base64.b64decode(frame_data)
        else:
            return None

        frame_pil = Image.open(BytesIO(frame_bytes))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


def process_tracking(frame, detector, tracker, args, frame_id):
    try:
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

        tracking_tlwhs = []
        tracking_ids = []
        tracking_scores = []
        tracking_bboxes = []

        if outputs is not None and len(outputs) > 0:
            online_targets = tracker.update(
                outputs, [img_info["height"], img_info["width"]], (128, 128)
            )

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id

                # Check if aspect ratio is valid
                if tlwh[2] > 0 and tlwh[3] > 0:  # Ensure width and height are positive
                    aspect_ratio = tlwh[2] / tlwh[3]
                    vertical = aspect_ratio > args["aspect_ratio_thresh"]

                    if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                        x1, y1, w, h = tlwh
                        tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                        tracking_tlwhs.append(tlwh)
                        tracking_ids.append(tid)
                        tracking_scores.append(t.score)

            tracking_image = plot_tracking(
                img_info["raw_img"],
                tracking_tlwhs,
                tracking_ids,
                names=id_face_mapping,
                frame_id=frame_id + 1,
            )
        else:
            tracking_image = img_info["raw_img"]

        return tracking_image, tracking_ids, bboxes, landmarks
    except Exception as e:
        print(f"Error in process_tracking: {e}")
        return None, [], [], []


@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    return emb_img_face / np.linalg.norm(emb_img_face)


def recognition(face_image):
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    return float(score[0]), name  # Ensure score is a Python float, not numpy float


def recognize_and_track(frame_data):
    try:
        # Check if frame_data is valid
        if not frame_data:
            return []

        # Process the frame
        frame = process_frame(frame_data)
        if frame is None:
            return []

        # Initialize tracker
        tracker = BYTETracker(args=config_tracking, frame_rate=30)

        # Process tracking
        tracking_image, tracking_ids, detection_bboxes, detection_landmarks = process_tracking(
            frame, detector, tracker, config_tracking, 0
        )

        # If no faces detected, return empty list
        # if not tracking_ids or (isinstance(detection_landmarks, np.ndarray) and detection_landmarks.size == 0):
        #     return []

        # Jika tidak ada wajah yang terdeteksi, kembalikan list kosong
        if not tracking_ids or len(detection_landmarks) == 0:
            return []

        # Process face recognition for all IDs
        detected_faces = []
        for i, tid in enumerate(tracking_ids):
            if i < len(detection_landmarks):  # Ensure we don't exceed landmark array bounds
                face_alignment = norm_crop(img=frame, landmark=detection_landmarks[i])
                score, name = recognition(face_alignment)

                # Convert numpy float to Python float for JSON serialization
                if isinstance(score, np.float32):
                    score = float(score)

                detected_faces.append(name if score >= 0.5 else 'unknown')

        return detected_faces

    except Exception as e:
        print(f"Error in recognize_and_track: {e}")
        return []


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker")
    mqtt.subscribe(FRAME_TOPIC)
    print(f"Subscribed to {FRAME_TOPIC}")


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    try:
        topic = message.topic
        payload = message.payload.decode('utf-8')

        if topic == FRAME_TOPIC:
            # Parse the payload
            data = json.loads(payload)

            # Process the frame and get results
            result = recognize_and_track(data)

            # Publish results
            if result:  # Only publish if we have results
                mqtt.publish(RESULT_TOPIC, json.dumps(result))
                print(f"Published result: {result}")

    except json.JSONDecodeError as e:
        print(f'Error decoding JSON from topic {topic}: {e}')
    except Exception as e:
        print(f'Error processing message from {topic}: {e}')


def main():
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()