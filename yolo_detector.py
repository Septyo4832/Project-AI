"""
yolo_detector.py
"""

from ultralytics import YOLO


class YOLODetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame):

        result = self.model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        return result[0]

    def get_detected_classes(self, result):

        detected_classes = []

        if result.boxes is None:
            return detected_classes

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            detected_classes.append(class_name)

        return detected_classes