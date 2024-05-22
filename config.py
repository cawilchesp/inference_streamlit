from ultralytics import YOLO
import streamlit as st
from pathlib import Path
import sys


ROOT = Path('D:/Data')

# DL model config
DETECTION_MODEL_DIR = ROOT / 'models' / 'yolov8'

MODEL_LIST = [
    'Nano',
    'Small',
    'Medium',
    'Large',
    'Extralarge' ]

DETECTION_MODEL_DICT = {
    'Nano': 'yolov8n.pt',
    'Small': 'yolov8s.pt',
    'Medium': 'yolov8m.pt',
    'Large': 'yolov8l.pt',
    'Extralarge': 'yolov8x.pt'
}

# Pose model config
POSE_MODEL_DICT = {
    'Nano': 'yolov8n-pose.pt',
    'Small': 'yolov8s-pose.pt',
    'Medium': 'yolov8m-pose.pt',
    'Large': 'yolov8l-pose.pt',
    'Extralarge': 'yolov8x-pose.pt'
}


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model