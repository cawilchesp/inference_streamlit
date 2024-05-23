from ultralytics import YOLO, RTDETR
import supervision as sv
import streamlit as st

import cv2
import sys
import torch
import signal
import tempfile
import itertools
import numpy as np
from typing import List
from pathlib import Path
from typing import Union, Optional, List
from functools import partial
from datetime import datetime

from sinks.model_sink import ModelSink
from sinks.track_sink import TrackSink
from sinks.annotate_sink import AnnotateSink
from sinks.region_sink import RegionSink
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import multi_sink

import config
from tools.general import find_in_list, load_zones
from tools.timers import ClockBasedTimer
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path, from_camera, VideoInfo
from tools.write_data import CSVSave

# For debugging
from icecream import ic

global PIPELINE
PIPELINE: Optional[InferencePipeline] = None
def signal_handler(sig, frame):
    print("Terminating")
    if PIPELINE is not None:
        PIPELINE.terminate()
        PIPELINE.join()
    sys.exit(0)


########################################
#                 Page
########################################
st.set_page_config(layout="wide", page_icon="🎥")

st.title('Video Detection')

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
    model_path = Path(config.DETECTION_MODEL_DIR, config.POSE_MODEL_DICT[model_type])
else:
    st.error('Select Model in Sidebar')

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

if source and zone_path and output:
    step_count = itertools.count(1)
    col_play, col_stop, col3 = st.columns([1, 1, 5])
    with col_play:
        play_button = st.button(label="Play Video")
    with col_stop:
        stop_button = st.button(label="Stop Video")

    video_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): quit()
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        source_info = from_camera(cap)
    else:
        source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Origen del Video Inicializado')
    print_video_info(source, source_info)

    width_text.write(f"{source_info.width} px")
    height_text.write(f"{source_info.height} px")
    total_frames_text.write(str(source_info.total_frames))
    fps_text.write(f"{source_info.fps:.2f} FPS")

    csv_writer = CSVSave(file_name=f"{output}.csv")
    video_writer = cv2.VideoWriter(
        filename=f"{output}.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=source_info.fps,
        frameSize=source_info.resolution_wh,
    )





    # Initialize model
    model_sink = ModelSink(
        weights_path=model_path,
        image_size=640,
        confidence=confidence )

    # Start video processing pipeline
    region_sink = RegionSink(
        source_info=source_info,
        track_length=50,
        iou=0.7,
        zone_configuration_path=zone_path,
        csv_writer=csv_writer,
        video_writer=video_writer,
        st_frame=st_frame,
        frame_text=frame_text,
        current_fps_text=current_fps_text )

    play_flag = False
    end_flag = False
    if play_button:
        play_flag = True
        end_flag = False
    if stop_button:
        play_flag = False
        end_flag = True
    with st.spinner("Running..."):
        try:
            if play_flag:
                PIPELINE = InferencePipeline.init_with_custom_logic(
                    video_reference=source,
                    on_video_frame=model_sink.detect,
                    on_prediction=region_sink.on_prediction,
                )
                PIPELINE.start()
            if end_flag:
                print("Terminating")
                video_writer.release()
                if PIPELINE is not None:
                    PIPELINE.terminate()
                    PIPELINE.join()
                sys.exit(0)

            







            # frame_number = 0
            # while play_flag:
            #     success, image = cap.read()
            #     if not success: break
            #     frame_text.write(str(frame_number))
            #     time_text.write(f"{frame_number / fps:.2f} s")
            #     # Resize the image to a standard size
            #     image = cv2.resize(image, (720, int(720 * (9 / 16))))

            #     results = model.track(
            #         source=image,
            #         persist=True,
            #         imgsz=640,
            #         conf=confidence,
            #         device='cuda' if torch.cuda.is_available() else 'cpu',
            #         retina_masks=True,
            #         verbose=False
            #     )[0]
            #     detections = sv.Detections.from_ultralytics(results)
            #     detections = detections.with_nms()

            #     # Draw labels
            #     object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
            #     annotated_image = label_annotator.annotate(
            #         scene=image,
            #         detections=detections,
            #         labels=object_labels )
                
            #     # Draw boxes
            #     annotated_image = bounding_box_annotator.annotate(
            #         scene=annotated_image,
            #         detections=detections )
                
            #     # Draw tracks
            #     if detections.tracker_id is not None:
            #         annotated_image = trace_annotator.annotate(
            #             scene=annotated_image,
            #             detections=detections )

            #     st_frame.image(
            #         annotated_image,
            #         caption='Detected Video',
            #         channels="BGR",
            #         use_column_width=True
            #     )
            #     frame_number += 1
            # cap.release()
        except Exception as e:
            st.error(f"Error loading video: {e}")
    