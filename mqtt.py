from flask import Flask
from flask_cors import CORS
import paho.mqtt.client as mqtt
import json
import numpy as np
import cv2
import base64
from datetime import datetime
import threading
import queue

app = Flask(__name__)
CORS(app)

# Konfigurasi MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_RECEIVE = "video/frames"
MQTT_TOPIC_SEND = "video/analysis"

# Queue untuk komunikasi antar thread
frame_queue = queue.Queue(maxsize=100)


def process_frame(frame_data):
    try:
        # Decode base64 frame
        img_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # analisis video sederhana
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Di sini bisa ditambahkan logika analisis video lainnya
        # Contoh: deteksi objek, analisis gerakan, dll.
        # frame = detector.detect_and_annotate(frame)

        return {
            "status": "success",
            "timestamp": str(datetime.now()),
            "analysis_result": "Frame processed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": str(datetime.now()),
            "error": str(e)
        }


# Callback ketika koneksi ke MQTT broker berhasil
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC_RECEIVE)


# Callback ketika menerima pesan MQTT
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        frame_queue.put(payload)
    except Exception as e:
        print(f"Error processing MQTT message: {e}")


# Thread untuk memproses frame
def process_frames(mqtt_client):
    while True:
        try:
            payload = frame_queue.get()
            frame_data = payload.get('frame')

            if frame_data:
                result = process_frame(frame_data)
                # Tambahkan timestamp original untuk tracking
                result['original_timestamp'] = payload.get('timestamp')

                # Kirim hasil analisis melalui MQTT
                mqtt_client.publish(MQTT_TOPIC_SEND, json.dumps(result), qos=1)

            frame_queue.task_done()
        except Exception as e:
            print(f"Error in processing thread: {e}")


def start_mqtt_client():
    # Setup MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect ke broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        return None

    return client


def main():
    # Start MQTT client
    mqtt_client = start_mqtt_client()
    if not mqtt_client:
        return

    # Start processing thread
    process_thread = threading.Thread(
        target=process_frames,
        args=(mqtt_client,),
        daemon=True
    )
    process_thread.start()

    # Start MQTT loop dalam thread terpisah
    mqtt_client.loop_start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000)

    # Cleanup
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


if __name__ == '__main__':
    main()