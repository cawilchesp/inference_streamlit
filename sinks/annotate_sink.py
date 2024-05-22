import supervision as sv

from inference.core.interfaces.camera.entities import VideoFrame

import cv2

from tools.video_info import VideoInfo
from tools.write_data import CSVSave




class AnnotateSink:
    def __init__(
        self,
        source_info: VideoInfo,
        track_length: int = 50,
        fps: bool = True,
        label: bool = True,
        box: bool = True,
        trace: bool = False
    ) -> None:
        self.fps = fps
        self.label = label
        self.box = box
        self.trace = trace
        
        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5
        
        if self.fps: self.fps_monitor = sv.FPSMonitor()
        
        if self.label: self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        if self.box: self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
        if self.trace: self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)
                

    def on_prediction(self, detections, frame: VideoFrame) -> None:
        if self.fps:
            self.fps_monitor.tick()
            fps_value = self.fps_monitor.fps

            annotated_image = frame.image.copy()
            annotated_image = sv.draw_text(
                scene=annotated_image,
                text=f"{fps_value:.1f}",
                text_anchor=sv.Point(40, 30),
                background_color=sv.Color.from_hex("#A351FB"),
                text_color=sv.Color.from_hex("#000000"),
            )

        if self.label:
            if detections.tracker_id is None:
                object_labels = [
                    f"{data['class_name']} ({score:.2f})"
                    for _, _, score, _, _, data in detections
                ]
            else:
                object_labels = [
                    f"{data['class_name']} {tracker_id} ({score:.2f})"
                    for _, _, score, _, tracker_id, data in detections
                ]
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=object_labels )
            
        if self.box:
            # Draw boxes
            annotated_image = self.bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
        if self.trace and detections.tracker_id is not None:
            # Draw tracks
            annotated_image = self.trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
        cv2.imshow("Processed Video", annotated_image)
        cv2.waitKey(1)
        
        return annotated_image, detections