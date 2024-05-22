from ultralytics import YOLO
import supervision as sv

import cv2
import torch
from tqdm import tqdm
from pathlib import Path
import itertools

from imutils.video import FPS

import config
from tools.print_info import print_video_info, step_message
from tools.video_info import from_video_path


# For debugging
from icecream import ic


dataset_names= {
    0: 'bicycle',
    1: 'bus',
    2: 'car',
    3: 'motorbike',
    4: 'person',
    5: 'truck'
}

class_conversion = {
    0: 4,
    1: 0,
    2: 2,
    3: 3,
    5: 1,
    7: 5
}

yolo_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def main(
    source: str,
    output: str,
    weights: str,
    image_size: int,
    confidence: float,
    class_filter: list[int],
    sample_number: int,
    show_image: bool
) -> None:
    step_count = itertools.count(1)

    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized")

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): quit()
    source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized')
    print_video_info(source, source_info)

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        k = int(scaled_width * source_info.height / source_info.width)
        scaled_height = k if source_info.height > k else source_info.height

    # Autolabelling settings
    stride = round(source_info.total_frames / sample_number)
    frame_generator = sv.get_video_frames_generator(source_path=source, stride=stride)
    
    # Annotators
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    image_sink = sv.ImageSink(target_dir_path=output)


    
    # image_count = 0
    total = round(source_info.total_frames / stride)
    fps = FPS().start()
    with image_sink:
        for image in tqdm(frame_generator, total=total, unit='frames'):
            txt_name = Path(image_sink.image_name_pattern.format(image_sink.image_count)).stem
            image_sink.save_image(image=image)

            annotated_image = image.copy()

            # Run YOLO inference
            results = model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                classes=class_filter,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )[0]
            if image_sink.image_count == 1:
                for values in dataset_names.values():
                    with open(f"{output}/labelmap.txt", 'a') as txt_file:
                        txt_file.write(f"{values}\n")

            for cls, xywhn in zip(results.boxes.cls.cpu().numpy(), results.boxes.xywhn.cpu().numpy()):
                center_x, center_y, width, height = xywhn
                new_cls = class_conversion[cls]
                with open(f"{output}/{txt_name}.txt", 'a') as txt_file:
                    txt_file.write(f"{int(new_cls)} {center_x} {center_y} {width} {height}\n")
            
            if show_image:
                detections = sv.Detections.from_ultralytics(results)
                
                # Draw labels
                object_labels = [f"{data['class_name']} ({score:.2f})" for _, _, score, _, _, data in detections]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=object_labels )
                
                # Draw boxes
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )

                # View live results
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                
                if show_image:
                    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
                    cv2.imshow("Output", annotated_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n")
                        break

                fps.update()

    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_dataset",
        weights=f"{config.YOLOV9_FOLDER}/{config.YOLOV9_WEIGHTS}.pt",
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        class_filter=config.CLASS_FILTER,
        sample_number=config.SAMPLE_NUMBER,
        show_image=config.SHOW_IMAGE
    )
