import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from icecream import ic


def draw_info_rectangle(scene, texts):
    shapes = np.zeros_like(scene, np.uint8)

    text_scale = 0.5
    text_thickness = 1
    text_fontface = cv2.FONT_HERSHEY_DUPLEX

    max_width = 0
    for text in texts:
        text_size = cv2.getTextSize(text, text_fontface, text_scale, text_thickness)[0]
        max_width = max(max_width, text_size[0])

    h = scene.shape[0]
    for text in texts:
        text_size = cv2.getTextSize(text, text_fontface, text_scale, text_thickness)[0]
        cv2.rectangle(
            img=shapes,
            pt1=(0, h),
            pt2=(max_width + 10, h - text_size[1] - 10),
            color=(255,255,255),
            thickness=cv2.FILLED )
        h = h - text_size[1] - 10
        
    out = scene.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(
        src1=scene,
        alpha=0.5,
        src2=shapes,
        beta=0.5,
        gamma=0 )[mask]
    
    h = scene.shape[0]
    for text in texts:
        text_size = cv2.getTextSize(text, text_fontface, text_scale, text_thickness)[0]
        cv2.putText(
            img=out,
            text=text,
            org=(5, h - 5),
            fontFace=text_fontface,
            fontScale=text_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA )
        h = h - text_size[1] - 10
    
    return out


def draw_time_rectangle(scene, time_text):
    shapes = np.zeros_like(scene, np.uint8)
    
    text_scale = 0.5
    text_thickness = 1
    text_fontface = cv2.FONT_HERSHEY_DUPLEX
    text_time_size = cv2.getTextSize(time_text, text_fontface, text_scale, text_thickness)[0]
    
    cv2.rectangle(
        img=shapes,
        pt1=(0, 0),
        pt2=(text_time_size[0] + 20, text_time_size[1] + 20),
        color=(255,255,255),
        thickness=cv2.FILLED )
    out = scene.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(
        src1=scene,
        alpha=0.5,
        src2=shapes,
        beta=0.5,
        gamma=0 )[mask]
    
    cv2.putText(
        img=out,
        text=time_text,
        org=(10, text_time_size[1] + 10),
        fontFace=text_fontface,
        fontScale=text_scale,
        color=(0, 0, 0),
        thickness=text_thickness,
        lineType=cv2.LINE_AA )
    
    return out


def line_cross(ultralytics_results, line, interest_keypoints, direction):
    if ultralytics_results.keypoints is None:
        return False
    
    x1, y1, x2, y2 = line
    if x1 == x2: 
        line_threshold = x1
    elif y1 == y2:
        line_threshold = y1

    crossed_count = 0
    for object in ultralytics_results:
        object_keypoints = object.keypoints.data.cpu().numpy()[0]
        track_id = object.boxes.id.int().cpu().numpy()[0] if object.boxes.id is not None else None
        for point_index in interest_keypoints:
            x, y, score = object_keypoints[point_index]
            if score > 0.5:
                if direction == 'right':
                    crossed_count = crossed_count + 1 if x > line_threshold else crossed_count
                elif direction == 'left':
                    crossed_count = crossed_count + 1 if x < line_threshold else crossed_count

    return True if crossed_count > 0 else False


def draw_line_cross(scene, line, crossed_flag):
    x1, y1, x2, y2 = line

    if crossed_flag:
        line_color = (0,0,255)
        text_line_crossed = 'X'
    else:
        line_color = (0,255,0)
        text_line_crossed = 'O'

    cv2.line(
        img=scene,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=line_color,
        thickness=2,
        lineType=cv2.LINE_AA )
    
    text_size = cv2.getTextSize(text_line_crossed, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    cv2.rectangle(
        img=scene,
        pt1=(int(x1 - (text_size[0]/2) - 5), 0),
        pt2=(int(x1 + (text_size[0]/2) + 5), text_size[1] + 10),
        color=line_color,
        thickness=-1 )
    
    cv2.putText(
        img=scene,
        text=text_line_crossed,
        org=(int(x1 - (text_size[0]/2)), text_size[1] + 5),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.7,
        color=(0, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA )
    
    return scene


def region_cross(ultralytics_results, region):
    if ultralytics_results.keypoints is None:
        return False
    
    in_count = 0
    for object in ultralytics_results:
        object_keypoints = object.keypoints.data.cpu().numpy()[0]
        track_id = object.boxes.id.int().cpu().numpy()[0] if object.boxes.id is not None else None
        for keypoint in object_keypoints:
            x, y, score = keypoint
            if score > 0.5:
                is_inside = Polygon(region).contains(Point(x, y))
                if is_inside:
                    in_count += 1

    return True if in_count > 0 else False


def draw_region_cross(scene, region, zone_flag):
    center_x = int((max(region[0]) + min(region)[0])/2)
    min_y = min(region)[1]

    if zone_flag:
        region_color = (0,0,255)
        text_region_crossed = 'X'
    else:
        region_color = (0,255,0)
        text_region_crossed = 'O'

    cv2.polylines(
        img=scene,
        pts=[np.array(region, dtype=np.int32)],
        isClosed=True,
        color=region_color,
        thickness=3 )
    
    text_size = cv2.getTextSize(text_region_crossed, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    cv2.rectangle(
        img=scene,
        pt1=(int(center_x - (text_size[0]/2) - 5), min_y),
        pt2=(int(center_x + (text_size[0]/2) + 5), min_y + text_size[1] + 10),
        color=region_color,
        thickness=-1 )
    
    cv2.putText(
        img=scene,
        text=text_region_crossed,
        org=(int(center_x - (text_size[0]/2)), min_y + text_size[1] + 5),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.7,
        color=(0, 0, 0),
        thickness=2 )
    
    return scene


def draw_alert(scene):   
    region_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
    shapes = np.zeros_like(scene, np.uint8)
    cv2.polylines(
        img=shapes,
        pts=[np.array(region_points, dtype=np.int32)],
        isClosed=True,
        color=(0,0,255),
        thickness=500 )
    out = scene.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(
        src1=scene,
        alpha=0.3,
        src2=shapes,
        beta=0.7,
        gamma=0 )[mask]
    
    return out