import supervision as sv

import cv2
import numpy as np
from datetime import datetime

from inference.core.interfaces.camera.entities import VideoFrame

from tools.general import load_zones
from tools.video_info import VideoInfo
from tools.timers import ClockBasedTimer
from tools.write_data import CSVSave

# For debugging
from icecream import ic


class RegionSink:
    def __init__(
        self,
        source_info: VideoInfo,
        track_length: int,
        iou: float,
        zone_configuration_path: str,
        csv_writer,
        video_writer,
        st_frame,
        frame_text,
        current_fps_text,
    ) -> None:
        self.tracker = sv.ByteTrack(minimum_matching_threshold=iou)

        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5
        
        self.COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness, color=self.COLORS)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness, color=self.COLORS)
        self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness, color=self.COLORS)
                
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones(file_path=zone_configuration_path)
        self.timers = [
            ClockBasedTimer()
            for _ in self.polygons
        ]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]

        self.csv_writer = csv_writer
        self.video_writer = video_writer

        self.st_frame = st_frame
        self.frame_text = frame_text
        self.current_fps_text = current_fps_text

    def on_prediction(self, detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        frame_number = frame.frame_id

        self.frame_text.write(str(frame_number))

        detections = self.tracker.update_with_detections(detections)

        self.current_fps_text.write(f"{fps:.1f} FPS")
        
        annotated_image = cv2.resize(frame.image, (720, int(720 * (9 / 16))))

        for idx, zone in enumerate(self.zones):
            annotated_image = sv.draw_polygon(
                scene=annotated_image,
                polygon=zone.polygon,
                color=self.COLORS.by_idx(idx)
            )

            if detections.tracker_id is not None:
                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = self.timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                object_labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    labels=object_labels,
                    custom_color_lookup=custom_color_lookup
                )
                
                # Draw boxes
                annotated_image = self.bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup
                )
                
                # Draw tracks
                annotated_image = self.trace_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup
                )

                custom_data = {
                    'frame_number': frame_number,
                    'time': datetime.now(),
                    'zone': idx
                }
                self.csv_writer.append(detections_in_zone, custom_data)
        self.video_writer.write(image=annotated_image)

        self.st_frame.image(
            annotated_image,
            caption='Detected Video',
            channels="BGR",
            use_column_width=True
        )