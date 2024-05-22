from ultralytics import YOLO
import supervision as sv

from inference.core.interfaces.camera.entities import VideoFrame

import torch
from typing import List


class ModelSink:
    def __init__(
        self,
        weights_path: str,
        image_size: int = 640,
        confidence: float = 0.5,
        class_filter: List[int] = None,
    ) -> None:
        self.model = YOLO(weights_path)
        self.image_size = image_size        
        self.confidence = confidence
        self.class_filter = class_filter

    def detect(self, frame: VideoFrame) -> sv.Detections:
        results = self.model(
            source=frame[0].image,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self.class_filter,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        detections = [sv.Detections.from_ultralytics(results).with_nms(threshold=0.7)]
        return detections