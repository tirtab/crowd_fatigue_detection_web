import logging

import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import time
import numpy as np

logging.basicConfig(level=logging.DEBUG)  # Atur level ke DEBUG untuk detail lebih lengkap


class YOLOv11FatigueDetector:
    def _init_(self):
        self.frame_width = 640
        self.frame_height = 480

        # Load model OpenVINO
        det_model_path = Path("model/fatigue_6_openvino_model/best.xml")
        core = ov.Core()
        det_ov_model = core.read_model(det_model_path)

        # Konfigurasi perangkat
        self.device = "AUTO"
        ov_config = {}

        if self.device != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in self.device or ("AUTO" in self.device and "GPU" in core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        det_compiled_model = core.compile_model(det_ov_model, self.device, ov_config)

        self.det_model = YOLO(det_model_path.parent, task="detect")

        if self.det_model.predictor is None:
            custom = {"conf": 0.5, "batch": 1, "save": False, "mode": "predict"}  # method defaults
            args2 = {**self.det_model.overrides, **custom}
            self.det_model.predictor = self.det_model._smart_load("predictor")(overrides=args2,
                                                                               _callbacks=self.det_model.callbacks)
            self.det_model.predictor.setup_model(model=self.det_model.model)

        self.det_model.predictor.model.ov_compiled_model = det_compiled_model

        # Inisialisasi anotator
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()

        # Pelacakan waktu untuk close_eye dan open_mouth
        self.close_eye_start_time = 0
        self.open_mouth_start_time = 0
        self.is_close_eye = False
        self.is_open_mouth = False

        print("Model Fatigue Detector berhasil dimuat")

    def detect_and_annotate(self, frame):
        try:
            result = self.det_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result).with_nms().with_nmm()
            detections = detections[detections.confidence > 0.5]

            labels = [
                f"{class_name} {confidence: .2f}"
                for class_name, confidence in zip(detections['class_name'], detections.confidence)
            ]

            logging.debug(f"Detected Class: {labels}")

            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            print(detections)
            return detections
        except Exception as e:
            logging.error(f"Error dalam anotasi deteksi: {e}")
            return frame, [0, 0, 0, 0]

    def get_fatigue_category(self, detections, threshold=0.5):
        try:
            current_time = time.time()

            # Hitung jumlah deteksi kelas "closed_eye" dengan confidence > threshold
            close_eye_count = sum(
                confidence > threshold
                for class_name, confidence in zip(detections["class_name"], detections.confidence)
                if class_name == "closed_eye"
            )

            # Jika kelas "closed_eye" terdeteksi minimal 2 kali
            if close_eye_count >= 2:
                if not self.is_close_eye:
                    self.close_eye_start_time = current_time
                    self.is_close_eye = True
                # Cek durasi mata tertutup
                elif current_time - self.close_eye_start_time >= 3:
                    return "Fatigue Detected: Close Eye"
            else:
                self.is_close_eye = False

            # Deteksi mulut terbuka
            open_mouth_score = max(
                confidence
                for class_name, confidence in zip(detections["class_name"], detections.confidence)
                if class_name == "open_mouth"
            ) if "open_mouth" in detections["class_name"] else 0

            if open_mouth_score > threshold:
                if not self.is_open_mouth:
                    self.open_mouth_start_time = current_time
                    self.is_open_mouth = True
                # Cek durasi mulut terbuka
                elif current_time - self.open_mouth_start_time >= 2:
                    return "Fatigue Detected: Open Mouth"
            else:
                self.is_open_mouth = False

            if open_mouth_score > threshold and close_eye_count >= 2:
                if not self.is_open_mouth and self.is_close_eye:
                    self.open_mouth_start_time = current_time
                    self.close_eye_start_time = current_time
                    self.is_open_mouth = True
                    self.is_close_eye = True
                # Cek durasi mulut terbuka
                elif current_time - self.open_mouth_start_time and current_time - self.close_eye_start_time >= 2:
                    return "Fatigue Detected: Open Mouth & Close Eye"
            else:
                self.is_open_mouth = False
                self.is_close_eye = False

            return "Normal"
        except Exception as e:
            return f"Error: {e}"