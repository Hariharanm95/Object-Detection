import torch

def load_model():
    """Loads the YOLOv5 model from PyTorch Hub."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(model, image):
    """Performs object detection on an image using the YOLOv5 model."""
    results = model(image)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord