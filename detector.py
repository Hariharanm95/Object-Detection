import torch
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov5s'):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    def detect_objects(self, image):
        # Perform inference
        results = self.model(image)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
