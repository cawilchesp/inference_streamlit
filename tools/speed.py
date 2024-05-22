import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    
    """
    from collections import defaultdict, deque


    ZONE_ANALYSIS = np.array([[530,128], [800,128], [1255,673], [0,673]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 75
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )
    
    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))

    Example:

    points = view_transformer.transform_points(points=points).astype(int)
    
    object_labels =[]
    for tracker_id, [x, y] in zip(tracks.tracker_id, points):
        coordinates[tracker_id].append([frame_number, x, y])
        if len(coordinates[tracker_id]) < source_info.fps / 2:
            object_labels.append(f"#{tracker_id}")
        else:
            t_0, x_0, y_0 = coordinates[tracker_id][0]
            t_1, x_1, y_1 = coordinates[tracker_id][-1]

            distance = abs(np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2))
            time_diff = (t_1 - t_0) / source_info.fps

            speed = distance / time_diff * 3.6
            object_labels.append(f"#{tracker_id} {int(speed)} Km/h")
    
    """