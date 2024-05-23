from ultralytics import YOLO, RTDETR
import supervision as sv
import streamlit as st

import sys
import cv2
import yaml
import torch
import time
import tempfile
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime

from imutils.video import FileVideoStream
from imutils.video import FPS

import config
from tools.general import load_zones
from tools.timers import ClockBasedTimer
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path, from_camera, VideoInfo
from tools.write_data import CSVSave


# For debugging
from icecream import ic


def main():
    ########################################
    #                 Page
    ########################################
    st.set_page_config(layout="wide", page_icon="🎥")

    st.title('Video Detection')

    col_source, col_json, col_output = st.columns([2, 2, 3])
    with col_source:
        source = st.text_input(
            label="Enter video path..."
        )
    with col_json:
        zone_path = st.text_input(
            label="Enter JSON file path..."
        )
    with col_output:
        output = st.text_input(
            label="Enter output path..."
        )

    col_image, col_info_1, col_info_2 = st.columns([5, 1, 1])
    with col_image:
        st_frame = st.empty()
    with col_info_1:
        st.markdown('**Width**')
        st.markdown('**Height**')
        st.markdown('**Total Frames**')
        st.markdown('**Original Frame Rate**')
        st.markdown('**Frame**')
        st.markdown('**Time**')
        st.markdown('**Current Frame Rate**')
    with col_info_2:
        width_text = st.markdown('0 px')
        height_text = st.markdown('0 px')
        total_frames_text = st.markdown('0')
        fps_text = st.markdown('0 FPS')
        frame_text = st.markdown('0')
        time_text = st.markdown('0.00 s')
        current_fps_text = st.markdown('0 FPS')

    ########################################
    #                Sidebar
    ########################################
    st.sidebar.header("Model")

    model_type = st.sidebar.selectbox(
        "Select Model",
        config.MODEL_LIST
    )

    confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))

    model_path = ""
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, config.DETECTION_MODEL_DICT[model_type])
    else:
        st.error('Select Model in Sidebar')

    # load pretrained DL model
    try:
        model = config.load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {model_path}")


    # if source_video:
    if source and zone_path and output:
        
        stop_button = st.button(label='Stop')

        video_source = eval(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): quit()
        if source.isnumeric() or source.lower().startswith('rtsp://'):
            source_info = from_camera(cap)
        else:
            source_info = from_video_path(cap)
        
        width_text.write(f"{source_info.width} px")
        height_text.write(f"{source_info.height} px")
        total_frames_text.write(str(source_info.total_frames))
        fps_text.write(f"{source_info.fps:.2f} FPS")

        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

        COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
        label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness, color=COLORS)
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness, color=COLORS)
        trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness, color=COLORS)
                
        fps_monitor = sv.FPSMonitor()
        polygons = load_zones(file_path=zone_path)
        timers = [
            ClockBasedTimer()
            for _ in polygons
        ]
        zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in polygons
        ]

        csv_writer = CSVSave(file_name=f"{output}.csv")
        video_writer = cv2.VideoWriter(
            filename=f"{output}.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=source_info.fps,
            frameSize=(720, int(720 * 9 / 16)),
        )

        tracker = sv.ByteTrack(minimum_matching_threshold=0.7)
        tracker.reset()

        frame_number = 0
        while cap.isOpened():
            fps_monitor.tick()
            fps_value = fps_monitor.fps

            success, image = cap.read()
            if not success: break

            frame_text.write(str(frame_number))
            time_text.write(f"{frame_number / source_info.fps:.2f} s")
            current_fps_text.write(f"{fps_value:.1f} FPS")

            # Resize the image to a standard size
            annotated_image = cv2.resize(image, (720, int(720 * 9 / 16)))

            results = model(
                source=annotated_image,
                imgsz=640,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                retina_masks=True,
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results).with_nms()
            detections = tracker.update_with_detections(detections)

            for idx, zone in enumerate(zones):
                annotated_image = sv.draw_polygon(
                    scene=annotated_image,
                    polygon=zone.polygon,
                    color=COLORS.by_idx(idx)
                )

                if detections.tracker_id is not None:
                    detections_in_zone = detections[zone.trigger(detections)]
                    time_in_zone = timers[idx].tick(detections_in_zone)
                    custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                    object_labels = [
                        f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                        for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                    ]

                    # Draw labels
                    # object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
                        detections=detections_in_zone,
                        labels=object_labels,
                        custom_color_lookup=custom_color_lookup
                    )
                    
                    # Draw boxes
                    annotated_image = bounding_box_annotator.annotate(
                        scene=annotated_image,
                        detections=detections_in_zone,
                        custom_color_lookup=custom_color_lookup
                    )
                    
                    # Draw tracks
                    annotated_image = trace_annotator.annotate(
                        scene=annotated_image,
                        detections=detections_in_zone,
                        custom_color_lookup=custom_color_lookup
                    )

                    custom_data = {
                        'frame_number': frame_number,
                        'time': datetime.now(),
                        'zone': idx
                    }
                    csv_writer.append(detections_in_zone, custom_data)
            video_writer.write(annotated_image)
            frame_number += 1

            st_frame.image(
                annotated_image,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
            )

            if stop_button:
                break
        sys.exit(0)
        # tracker.reset()
        # cap.release()
        # csv_writer.file.close()
        # video_writer.release()


if __name__ == '__main__':
    main()