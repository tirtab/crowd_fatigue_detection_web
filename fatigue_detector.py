import logging

import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import time
import numpy as np

logging.basicConfig(level=logging.DEBUG)  # Atur level ke DEBUG untuk detail lebih lengkap


class YOLOv11FatigueDetector:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(YOLOv11FatigueDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Cegah inisialisasi ulang
        if hasattr(self, 'initialized'):
            return

        try:
            # Logging yang lebih detail
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

            # Optimasi konfigurasi model
            self.frame_width = 640
            self.frame_height = 480

            # Load model dengan error handling yang lebih baik
            det_model_path = Path("model/fatigue_6_openvino_model/best.xml")
            if not det_model_path.exists():
                raise FileNotFoundError(f"Model not found at {det_model_path}")

            core = ov.Core()
            det_ov_model = core.read_model(det_model_path)

            # Konfigurasi perangkat dengan lebih fleksibel
            self.device = self._select_optimal_device(core)
            det_compiled_model = self._compile_model(core, det_ov_model)

            self.det_model = YOLO(det_model_path.parent, task="detect")
            self._setup_predictor(det_compiled_model)

            # Inisialisasi annotator dengan konfigurasi yang dapat disesuaikan
            self.box_annotator = sv.BoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
            self.label_annotator = sv.LabelAnnotator()

            self.initialized = True
            self.logger.info("Fatigue Detector berhasil diinisialisasi")

        except Exception as e:
            self.logger.error(f"Inisialisasi Fatigue Detector gagal: {e}")
            raise

    def _select_optimal_device(self, core):
        """Pilih perangkat optimal untuk inferensi"""
        available_devices = core.available_devices
        self.logger.info(f"Perangkat tersedia: {available_devices}")

        # Prioritas: GPU > AUTO > CPU
        if "GPU" in available_devices:
            return "GPU"
        elif "AUTO" in available_devices:
            return "AUTO"
        return "CPU"

    def _compile_model(self, core, det_ov_model):
        """Kompilasi model dengan konfigurasi khusus"""
        ov_config = {}
        if self.device != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})

        if "GPU" in self.device:
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

        return core.compile_model(det_ov_model, self.device, ov_config)

    def _setup_predictor(self, compiled_model):
        """Konfigurasi prediktor dengan parameter optimal"""
        if self.det_model.predictor is None:
            custom = {
                "conf": 0.5,  # Threshold confidence
                "batch": 1,
                "save": False,
                "mode": "predict"
            }
            args2 = {**self.det_model.overrides, **custom}
            self.det_model.predictor = self.det_model._smart_load("predictor")(
                overrides=args2,
                _callbacks=self.det_model.callbacks
            )
            self.det_model.predictor.setup_model(model=self.det_model.model)

            self.det_model.predictor.model.ov_compiled_model = compiled_model

    def detect_and_annotate(self, frame):
        try:
            result = self.det_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result).with_nms().with_nmm()
            detections = detections[detections.confidence > 0.5]

            detected_classes = {class_name: confidence for class_name, confidence in
                                zip(detections['class_name'], detections.confidence)}

            class_scores = [
                detected_classes.get("closed_eye", 0),
                detected_classes.get("open_eye", 0),
                detected_classes.get("closed_mouth", 0),
                detected_classes.get("open_mouth", 0)
            ]

            logging.debug(f"Detected class_scores: {class_scores}")

            labels = [
                f"{class_name} {confidence: .2f}"
                for class_name, confidence in zip(detections['class_name'], detections.confidence)
            ]

            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            return frame, class_scores
        except Exception as e:
            logging.error(f"Error dalam anotasi deteksi: {e}")
            return frame, [0, 0, 0, 0]

    def get_fatigue_category(self, class_scores):
        """Algoritma deteksi kelelahan yang lebih kompleks"""
        thresholds = {
            'close_eye': 0.5,  # Threshold lebih tinggi
            'open_mouth': 0.5,
            'duration_close_eye': 3,  # Detik
            'duration_open_mouth': 2
        }

        try:
            current_time = time.time()

            # Deteksi mata tertutup
            if class_scores[0] > thresholds['close_eye']:
                if not self.is_close_eye:
                    self.close_eye_start_time = current_time
                    self.is_close_eye = True

                # Cek durasi mata tertutup
                if current_time - self.close_eye_start_time >= thresholds['duration_close_eye']:
                    return "Fatigue: Mata Tertutup Lama"
            else:
                self.is_close_eye = False

            # Deteksi mulut terbuka
            if class_scores[3] > thresholds['open_mouth']:
                if not self.is_open_mouth:
                    self.open_mouth_start_time = current_time
                    self.is_open_mouth = True

                # Cek durasi mulut terbuka
                if current_time - self.open_mouth_start_time >= thresholds['duration_open_mouth']:
                    return "Fatigue: Mulut Terbuka Lama"
            else:
                self.is_open_mouth = False

            return "Normal"

        except Exception as e:
            self.logger.error(f"Gagal mengecek kelelahan: {e}")
            return "Error"
