import cv2
import numpy as np
import supervision as sv
from numpy import ndarray
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import logging

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


class YOLOv11CrowdDetector:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(YOLOv11CrowdDetector, cls).__new__(cls)
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
            det_model_path = Path("model/crowd_openvino_model/best.xml")
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

            # Inisialisasi zona plygon dan annotator dengan konfigurasi yang dapat disesuaikan
            zone_polygon = (ZONE_POLYGON * np.array([self.frame_width, self.frame_height])).astype(int)
            self.zone = sv.PolygonZone(polygon=zone_polygon)
            self.zone_annotator = sv.PolygonZoneAnnotator(
                zone=self.zone,
                color=sv.Color.RED,
                thickness=2,
                text_thickness=4,
                text_scale=2,
            )

            self.initialized = True
            self.logger.info("Crowd Detector berhasil diinisialisasi")

        except Exception as e:
            self.logger.error(f"Inisialisasi Crowd Detector gagal: {e}")
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
        # Deteksi menggunakan YOLOv11
        # def callback(image_slice: np.ndarray) -> sv.Detections:
        #     result = self.det_model(image_slice)[0]
        #     return sv.Detections.from_ultralytics(result).with_nms().with_nmm()
        #
        # slicer = sv.InferenceSlicer(callback=callback,
        #                             overlap_ratio_wh=None, overlap_wh=(64, 64),
        #                             overlap_filter=sv.OverlapFilter.NON_MAX_MERGE,
        #                             iou_threshold=0.3)
        # detections = slicer(frame)
        try:
            # Pastikan frame memiliki ukuran yang benar
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # Deteksi dengan NMS
            result = self.det_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result).with_nms()
            detections = detections[detections.confidence > 0.5]

            # Anotasi bounding box dan label
            labels = [
                f"{class_name} {confidence: .2f}"
                for class_name, confidence
                in zip(detections.data.get("class_name", []), detections.confidence)
            ]

            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Anotasi zona
            self.zone.trigger(detections=detections)
            frame = self.zone_annotator.annotate(scene=frame)

            # Ekstraksi data deteksi dengan struktur yang lebih efisien
            detection_data = [{
                "class_name": detections.data.get("class_name", [])[i],
                "confidence": float(detections.confidence[i]),
                "bounding_box": {
                    "x_min": int(detections.xyxy[i][0]),
                    "y_min": int(detections.xyxy[i][1]),
                    "x_max": int(detections.xyxy[i][2]),
                    "y_max": int(detections.xyxy[i][3])
                }
            } for i in range(len(detections))]

            return frame, detection_data

        except Exception as e:
            self.logger.error(f"Deteksi crowd gagal: {e}")
            return frame, []

    @staticmethod
    def get_crowd_category(count):
        """Mengembalikan kategori berdasarkan jumlah deteksi."""
        if count > 20:
            return "Ramai"
        elif 10 <=  count <= 20:
            return "Sedang"
        else:
            return "Sedikit"
