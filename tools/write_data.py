from supervision.detection.core import Detections

import csv
from typing import Any, Dict, List, Optional

from icecream import ic


class CSVSave:
    def __init__(
        self,
        file_name: str = "output.csv"
    ) -> None:
        self.file_name = file_name
        self.file = open(self.file_name, "w", newline="")
        self.writer = csv.writer(self.file)
        
    def parse_detection_data(detections: Detections, custom_data: Dict[str, Any] = None):
        parsed_rows = []
        for xyxy, _, confidence, class_id, tracker_id, data in detections:

            row = {
                "x": str(int(xyxy[0])),
                "y": str(int(xyxy[1])),
                "w": str(int(xyxy[2] - xyxy[0])),
                "h": str(int(xyxy[3] - xyxy[1])),
                "confidence": "" if confidence is None else str(confidence),
                "class_id": "" if class_id is None else str(class_id),
                "tracker_id": "" if tracker_id is None else str(tracker_id),
                "class_name": "" if data is None else data['class_name'],
            }

            if custom_data:
                row.update(custom_data)

            parsed_rows.append(row)
            
        return parsed_rows
    
    def append(self, detections: Detections, custom_data: Dict[str, Any] = None) -> None:
        BASE_HEADER = [
            "tracker_id",
            "class_id",
            "x",
            "y",
            "w",
            "h",
            "confidence"]
        custom_headers = list(custom_data.keys())
        field_names = BASE_HEADER + custom_headers

        parsed_rows = CSVSave.parse_detection_data(detections, custom_data)
        for row in parsed_rows:
            self.writer.writerow([row.get(field_name, "") for field_name in field_names])
        