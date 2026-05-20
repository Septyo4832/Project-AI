"""
yolo_detector.py

TODO:
- Load model YOLO dari file best.pt.
- Jalankan deteksi pada frame kamera.
- Ambil nama class yang terdeteksi.
- Ubah hasil deteksi menjadi fakta untuk Production System.
"""

from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        """
        TODO:
        Load model YOLO.
        """

        self.model = YOLO(model_path)

        # TODO:
        # model.names berisi daftar class dari model
        # contoh:
        # {0: 'face_eyes_closed', 1: 'face_eyes_open', 2: 'head_down', 3: 'phone'}
        self.class_names = self.model.names

    def detect(self, frame):
        """
        TODO:
        Jalankan deteksi YOLO pada 1 frame kamera.
        """

        result = self.model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        # TODO:
        # Karena hasil predict berbentuk list, ambil index pertama
        return result[0]

    def get_detected_classes(self, result):
        """
        TODO:
        Ambil nama class yang terdeteksi dari hasil YOLO.
        """

        detected_classes = []

        # TODO:
        # Cek apakah ada bounding box yang terdeteksi
        if result.boxes is None:
            return detected_classes

        # TODO:
        # Loop semua box hasil deteksi
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            detected_classes.append(class_name)

        return detected_classes

    def make_facts(self, detected_classes, eyes_closed_duration):
        """
        TODO:
        Ubah daftar class YOLO menjadi fakta untuk Production System.
        """

        facts = {
            "face_detected": False,
            "eyes_open": False,
            "eyes_closed": False,
            "phone_detected": False,
            "head_down": False,
            "eyes_closed_duration": eyes_closed_duration
        }

        # TODO:
        # Sesuaikan nama class dengan model
        if "face_eyes_open" in detected_classes:
            facts["face_detected"] = True
            facts["eyes_open"] = True

        if "face_eyes_closed" in detected_classes:
            facts["face_detected"] = True
            facts["eyes_closed"] = True

        if "phone" in detected_classes:
            facts["phone_detected"] = True

        if "head_down" in detected_classes:
            facts["head_down"] = True

        return facts