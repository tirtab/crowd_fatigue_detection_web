from flask import Flask, render_template, Response
from yolov11_detector import YOLOv11CrowdDetector
import cv2
import logging
import paho.mqtt.client as mqtt
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Inisialisasi detektor dan kamera
try:
    detector = YOLOv11CrowdDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)
    detector = None

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logging.error("Kamera tidak dapat diakses")
    camera = None

# Inisialisasi MQTT client
mqtt_client = mqtt.Client("FlaskDetector")
mqtt_client.connect("localhost", 1883)  # Ganti "localhost" jika broker berada di alamat lain


def generate_frames():
    """Streaming video dari kamera dengan deteksi real-time."""
    if not camera or not detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak tersedia.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            frame, detection_data = detector.detect_and_annotate(frame)
            num_people = len(detection_data)
            category = detector.get_crowd_category(num_people)

            # Tambahkan data jumlah dan kategori ke dalam data deteksi
            detection_summary = {
                "num_people": num_people,
                "category": category,
                "detections": detection_data  # Tambahkan semua data bounding box dan confidence
            }

            # Publikasikan data deteksi ke MQTT
            mqtt_client.publish("crowd/detection", json.dumps(detection_summary))

            # Encoding frame untuk streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as error:
            logging.error("Error dalam memproses frame: %s", error)
            break


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
