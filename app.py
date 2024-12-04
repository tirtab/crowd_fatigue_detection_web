from flask import Flask, render_template, Response
from flask_mqtt import Mqtt
from crowd_detector import YOLOv11CrowdDetector
from fatigue_detector import YOLOv11FatigueDetector
import cv2
import logging
import json
from datetime import datetime
import base64
from PIL import Image
import numpy as np
from io import BytesIO

# Inisialisasi Flask dan Flask-MQTT
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Konfigurasi MQTT
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # Dalam detik

mqtt = Mqtt(app)

# Topik MQTT
CROWD_FRAME_TOPIC = 'mqtt-crowd-frame'
FATIGUE_FRAME_TOPIC = 'mqtt-fatigue-frame'
CROWD_RESULT_TOPIC = 'mqtt-crowd-result'
FATIGUE_RESULT_TOPIC = 'mqtt-fatigue-result'

# Variabel Global untuk Menyimpan Frame Terakhir
latest_crowd_frame = None
latest_fatigue_frame = None

# Inisialisasi Kamera dan Detektor
try:
    camera = cv2.VideoCapture(1)  # Gunakan indeks kamera yang sesuai
    if not camera.isOpened():
        logging.error("Kamera tidak dapat diakses.")
        camera = None
except Exception as e:
    logging.error(f"Gagal menginisialisasi kamera: {e}")
    camera = None

try:
    crowd_detector = YOLOv11CrowdDetector()
    fatigue_detector = YOLOv11FatigueDetector()
except Exception as e:
    logging.error(f"Gagal menginisialisasi detektor: {e}")
    crowd_detector = None
    fatigue_detector = None


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
    raise TypeError(f"Type {type(obj)} not serializable")


# Fungsi untuk Menghasilkan Streaming Crowd Analysis
def generate_crowd_frames():
    if not camera or not crowd_detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            frame, detection_data = crowd_detector.detect_and_annotate(frame)
            num_people = len(detection_data)

            # Publikasikan Hasil ke MQTT
            mqtt_data = {"status": "success",
                         "timestamp": str(datetime.now()),
                         "num_people": num_people,
                         "detections": detection_data}
            mqtt.publish(CROWD_RESULT_TOPIC, json.dumps(mqtt_data, default=custom_serializer))

            # Encode Frame untuk Streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error(f"Error dalam memproses frame crowd: {e}")
            break


# Fungsi untuk Menghasilkan Streaming Fatigue Analysis
def generate_fatigue_frames():
    if not camera or not fatigue_detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            frame, detected_classes = fatigue_detector.detect_and_annotate(frame)
            fatigue_status = fatigue_detector.get_fatigue_category(detected_classes)

            # Publikasikan Hasil ke MQTT
            mqtt_data = {"status": fatigue_status, "timestamp": str(datetime.now())}
            mqtt.publish(FATIGUE_RESULT_TOPIC, json.dumps(mqtt_data, default=custom_serializer))

            # Tambahkan Status ke Frame
            cv2.putText(frame, fatigue_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode Frame untuk Streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error(f"Error dalam memproses frame fatigue: {e}")
            break


# MQTT Event Handlers
@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker")
    mqtt.subscribe(CROWD_FRAME_TOPIC)
    mqtt.subscribe(FATIGUE_FRAME_TOPIC)
    print(f"Subscribed to {CROWD_FRAME_TOPIC} and {FATIGUE_FRAME_TOPIC}")


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    global latest_crowd_frame, latest_fatigue_frame
    topic = message.topic
    payload = message.payload.decode('utf-8')

    try:
        data = json.loads(payload)

        if topic == CROWD_FRAME_TOPIC:
            latest_crowd_frame = process_frame(data['frame'])
            if latest_crowd_frame is not None:
                crowd_result = {"status": "success",
                                "timestamp": str(datetime.now()),
                                "num_people": len(crowd_detector.detect_and_annotate(latest_crowd_frame)[1])}
                mqtt.publish(CROWD_RESULT_TOPIC, json.dumps(crowd_result))

        elif topic == FATIGUE_FRAME_TOPIC:
            latest_fatigue_frame = process_frame(data['frame'])
            if latest_fatigue_frame is not None:
                fatigue_result = {"status": fatigue_detector.get_fatigue_category(fatigue_detector.detect_and_annotate(latest_fatigue_frame)[1]),
                                 "timestamp": str(datetime.now())}
                mqtt.publish(FATIGUE_RESULT_TOPIC, json.dumps(fatigue_result))

    except Exception as e:
        logging.error(f"Error processing MQTT message: {e}")


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crowd_analysis')
def crowd_analysis():
    return render_template('crowd_analysis.html')


@app.route('/fatigue_analysis')
def fatigue_analysis():
    return render_template('fatigue_analysis.html')


@app.route('/video_feed/crowd')
def video_feed_crowd():
    return Response(generate_crowd_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/fatigue')
def video_feed_fatigue():
    return Response(generate_fatigue_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
