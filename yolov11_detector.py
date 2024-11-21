import numpy as np
import supervision as sv
from numpy import ndarray
from ultralytics import YOLO
from pathlib import Path
import openvino as ov

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


class YOLOv11CrowdDetector:
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480

        # Load model OpenVINO
        det_model_path = Path("model/best_openvino_model/best.xml")
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

        zone_polygon = (ZONE_POLYGON * np.array([self.frame_width, self.frame_height])).astype(int)
        self.zone = sv.PolygonZone(polygon=zone_polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone,
            color=sv.Color.RED,
            thickness=2,
            text_thickness=4,
            text_scale=2,
        )

    def detect_and_annotate(self, frame):
        # Deteksi menggunakan YOLOv11
        result = self.det_model(frame)[0]
        detections = sv.Detections.from_ultralytics(result).with_nms().with_nmm()
        detections = detections[detections.confidence > 0.5]

        # Anotasi bounding box dan label
        labels = [
            f"{class_name} {confidence: .2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Anotasi zona
        self.zone.trigger(detections=detections)
        frame: ndarray = self.zone_annotator.annotate(scene=frame)

        # Ekstrak data bounding box, class, dan confidence untuk setiap deteksi
        detection_data = []
        for detection in detections:
            box = detection[0]  # Asumsikan `box` menyimpan koordinat bounding box
            detection_data.append({
                "bounding_box": {
                    "x_min": int(box[0]),
                    "y_min": int(box[1]),
                    "x_max": int(box[2]),
                    "y_max": int(box[3])
                }
            })

        return frame, detection_data  # Kembalikan frame yang sudah dianotasi beserta data deteksi

    def get_crowd_category(self, count):
        """Mengembalikan kategori berdasarkan jumlah deteksi."""
        if count > 20:
            return "Ramai"
        elif 10 <= count <= 20:
            return "Sedang"
        else:
            return "Sedikit"
