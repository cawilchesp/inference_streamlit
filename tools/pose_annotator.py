import cv2
import numpy as np

from supervision.annotators.base import BaseAnnotator

from icecream import ic


class PoseAnnotator(BaseAnnotator):
    """ A class for drawing pose keypoints on an image using provided detections. """
    def __init__(
        self,
        thickness: int = 1,
        radius: int = 1,
        text_thickness: int = 1
    ):
        """
        Args:
            thickness (int): Thickness of the keypoint lines.
            radius (int): radius of the keypoints.
        """
        self.thickness: int = thickness
        self.radius: int = radius
        self.text_thickness: int = text_thickness
        self.color_list = [ (150,255,150), (255,150,150), (150,150,255), (0,255,255) ]
        self.keypoint_circles = [ (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1),
            (11, 2), (12, 2), (13, 3), (14, 3), (15, 3), (16, 3) ]
        self.keypoint_lines = [ (0,1,0), (0,2,0), (1,2,0), (1,3,0), (2,4,0),
            (5,7,1), (7,9,1), (5,6,1), (6,8,1), (8,10,1), (5,11,2),
            (6,12,2), (11,12,2), (11,13,3), (13,15,3), (12,14,3), (14,16,3) ]
        self.keypoint_names = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

    def annotate(
        self,
        scene: np.ndarray,
        ultralytics_results,
        circles: bool = True,
        lines: bool = True,
        labels: bool = False
    ) -> np.ndarray:
        """
        Annotates the given scene with pose keypoints based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            ultralytics_results: Object detections to annotate.
            circles (bool): Keypoint circles.
            lines (bool): Lines between keypoints.
            labels (bool): Keypoint labels with coordinates.

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            pose_annotator = PoseAnnotator()
            annotated_frame = pose_annotator.annotate(
                scene=image.copy(),
                ultralytics_results=results
            )
            ```
        """
        if ultralytics_results.keypoints is None:
            return scene
        
        for object in ultralytics_results:
            object_keypoints = object.keypoints.data.cpu().numpy()[0]
            
            # Draw pose lines
            if lines:
                for keypoint_line in self.keypoint_lines:
                    line1_index, line2_index, color_index = keypoint_line
                    x1, y1, score1 = object_keypoints[line1_index]
                    x2, y2, score2 = object_keypoints[line2_index]
                    if score1 > 0.5 and score2 > 0.5:
                        cv2.line(
                            img=scene,
                            pt1=(int(x1),int(y1)),
                            pt2=(int(x2),int(y2)),
                            color=self.color_list[color_index],
                            thickness=self.thickness,
                            lineType=cv2.LINE_AA )
                        
            # Draw pose points
            for keypoint_circle in self.keypoint_circles:
                index, color_index = keypoint_circle
                x, y, score = int(object_keypoints[index][0]), int(object_keypoints[index][1]), object_keypoints[index][2]
                if score > 0.5:
                    if circles:
                        cv2.circle(
                            img=scene,
                            center=(x, y),
                            radius=self.radius,
                            color=self.color_list[color_index],
                            thickness=cv2.FILLED )
                        
                    if labels:
                        point_text = f"({x}, {y})"
                        point_text_size = cv2.getTextSize(
                            text=point_text,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.25,
                            thickness=self.text_thickness )[0]
                        cv2.rectangle(
                            img=scene,
                            pt1=(x+2, y-4-point_text_size[1]),
                            pt2=(x+6+point_text_size[0], y),
                            color=self.color_list[color_index],
                            thickness=cv2.FILLED )
                        cv2.putText(
                            img=scene,
                            text=point_text,
                            org=(x+4, y-2),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.25,
                            color=(0,0,0),
                            thickness=self.text_thickness,
                            lineType=cv2.LINE_AA )

        return scene
    

    def track_point(
        self,
        scene: np.ndarray,
        ultralytics_results,
        frame_number: int,
        keypoint_track,
        keypoint: str = 'right_wrist'
    ) -> np.ndarray:
        """
        Annotates the given scene with keypoint trajectory based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            ultralytics_results: Object detections to annotate.
            hand_track: List of trajectory points
            hand (str): Selected hand

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            annotated_image = pose_annotator.track_point(
                scene=annotated_image,
                ultralytics_results=results,
                hand_track=hand_track,
                hand='left_wrist'
            )
            ```
        """
        if ultralytics_results.keypoints is None:
            return scene
        
        for object in ultralytics_results:
            object_keypoints = object.keypoints.data.cpu().numpy()[0]
            track_id = object.boxes.id.int().cpu().numpy()[0] if object.boxes.id is not None else None
        
            index = self.keypoint_names[keypoint]
            x, y, score = int(object_keypoints[index][0]), int(object_keypoints[index][1]), object_keypoints[index][2]
            if track_id and score > 0.5:
                keypoint_track[track_id].append((frame_number, x, y))

            # Draw hand trajectory
            if len(keypoint_track[track_id]) > 1:
                pts = np.array(list(keypoint_track[track_id]), np.int32)[:, 1:3].reshape((-1, 1, 2))
                cv2.polylines(
                    img=scene,
                    pts=np.int32([pts]),
                    isClosed=False,
                    color=(255, 255, 255),
                    thickness=self.thickness,
                    lineType=cv2.LINE_AA )

        return scene


    def angle_vector(
        self,
        pt1: tuple[int,int],
        pt2: tuple[int,int]
    ) -> float:
        """ Calculate angle between two points and the horizontal
        
        Args:
            pt1 (tuple[int,int]): First point of vector.
            pt2 (tuple[int,int]): Second point of vector.

        Returns:
            Angle in degrees (float).

        """
        line0 = np.cross([pt1[0], pt1[1], 1], [1000, pt2[1], 1])
        line1 = np.cross([pt1[0], pt1[1], 1], [pt2[0], pt2[1], 1])

        a1,b1,c1 = line0
        a2,b2,c2 = line1

        # intersection point
        src_pnt = np.cross([a1,b1,c1], [a2,b2,c2])

        if src_pnt[-1] != 0:
            src_pnt = np.int16(src_pnt / src_pnt[-1])

            # angle between two lines
            num = a1*b2 - b1*a2
            den = a1*a2 + b1*b2
            if den != 0:
                theta = abs(np.arctan(num/den))*180/3.1416
        else:
            theta = 0.0

        return theta
    

    def speed_vector(
        self,
        scene: np.ndarray,
        keypoint_track,
        vector_track
    ) -> np.ndarray:
        for track_id, points in keypoint_track.items():
            if len(points) > 1:
                t_0, x_0, y_0 = points[-2]
                t_1, x_1, y_1 = points[-1]
                distance = abs(np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2))
                time_diff = (t_1 - t_0)
                speed = distance / time_diff if time_diff != 0 else distance

                # angle
                pnt1 = (x_1, y_1)
                pnt2 = (2*x_1-x_0, 2*y_1-y_0)
                angle = self.angle_vector(
                    pt1=pnt1,
                    pt2=pnt2 )
            
                vector_track[track_id].append((t_0, speed, angle))
                
                cv2.arrowedLine(
                    img=scene,
                    pt1=(pnt1[0], pnt1[1]),
                    pt2=(pnt2[0], pnt2[1]),
                    color=(255,255,255),
                    thickness=2,
                    line_type=cv2.LINE_AA
                )
                cv2.circle(
                    img=scene,
                    center=(pnt1[0], pnt1[1]),
                    radius=2,
                    color=(255,255,255),
                    thickness=cv2.FILLED )
                cv2.line(
                    img=scene,
                    pt1=(pnt1[0]-50, pnt1[1]),
                    pt2=(pnt1[0]+50, pnt1[1]),
                    color=(255,255,255),
                    thickness=1,
                    lineType=cv2.LINE_AA )

        return scene
    

    def fall_detection(
        self,
        scene: np.ndarray,
        ultralytics_results,
    ) -> np.ndarray:
        """
        Annotates the given scene with keypoint trajectory based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            ultralytics_results: Object detections to annotate.
            hand_track: List of trajectory points
            hand (str): Selected hand

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            annotated_image = pose_annotator.track_point(
                scene=annotated_image,
                ultralytics_results=results,
                hand_track=hand_track,
                hand='left_wrist'
            )
            ```
        """
        if ultralytics_results.keypoints is None:
            return scene
        
        h, w, _ = scene.shape
        for object in ultralytics_results:
            object_keypoints = object.keypoints.data.cpu().numpy()[0]
            track_id = object.boxes.id.int().cpu().numpy()[0] if object.boxes.id is not None else None
        
            keypoint_array = np.array(object_keypoints)
            mask = keypoint_array[:,2] >= 0.5
            keypoint_array = keypoint_array[mask]

            person_height = max(keypoint_array[:,1]) - min(keypoint_array[:,1])
            percentage = person_height / h
            
            fall_flag = True if percentage < 0.4 else False
            color = (0,0,255) if percentage < 0.4 else (0,255,0)

            max_index = np.argmax(np.array(keypoint_array)[:,1])
            min_index = np.argmin(np.array(keypoint_array)[:,1])
            high_point = (int(np.array(keypoint_array)[max_index][0]), int(np.array(keypoint_array)[max_index][1]))
            low_point = (int(np.array(keypoint_array)[min_index][0]), int(np.array(keypoint_array)[min_index][1]))
            
            cv2.line(
                img=scene,
                pt1=high_point,
                pt2=low_point,
                color=color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA )

        return scene, fall_flag
    

    def hands_up(
        self,
        scene: np.ndarray,
        ultralytics_results,
    ) -> np.ndarray:
        """
        Annotates the given scene with keypoint trajectory based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            ultralytics_results: Object detections to annotate.
            hand_track: List of trajectory points
            hand (str): Selected hand

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            annotated_image = pose_annotator.track_point(
                scene=annotated_image,
                ultralytics_results=results,
                hand_track=hand_track,
                hand='left_wrist'
            )
            ```
        """
        if ultralytics_results.keypoints is None:
            return scene
        
        for object in ultralytics_results:
            object_keypoints = object.keypoints.data.cpu().numpy()[0]
            track_id = object.boxes.id.int().cpu().numpy()[0] if object.boxes.id is not None else None
        
            keypoint_array = np.array(object_keypoints)
            
            head_array = keypoint_array[0:5,:]
            head_mask = keypoint_array[0:5,2] >= 0.5
            head_keypoint_array = head_array[head_mask]
            if head_keypoint_array.size != 0:
                min_head_y = min(head_keypoint_array[:, 1])
                min_index = np.argmin(np.array(head_keypoint_array)[:, 1])
                min_head_keypoint = (int(np.array(head_keypoint_array)[min_index][0]), int(np.array(head_keypoint_array)[min_index][1]))
            
                if keypoint_array[9, 2] > 0.5 and keypoint_array[10, 2] > 0.5:
                    left_wrist_y = keypoint_array[9, 1]
                    right_wrist_y = keypoint_array[10, 1]
                    left_wrist_keypoint = (int(np.array(keypoint_array)[9][0]), int(np.array(keypoint_array)[9][1]))
                    right_wrist_keypoint = (int(np.array(keypoint_array)[10][0]), int(np.array(keypoint_array)[10][1]))

                    if left_wrist_y < min_head_y and right_wrist_y < min_head_y:
                        hands_up_flag = True
                        color = (0,255,0)
                    else:
                        hands_up_flag = False
                        color = (0,0,255)
                    
                    cv2.line(
                        img=scene,
                        pt1=min_head_keypoint,
                        pt2=left_wrist_keypoint,
                        color=color,
                        thickness=self.thickness,
                        lineType=cv2.LINE_AA )
                    
                    cv2.line(
                        img=scene,
                        pt1=min_head_keypoint,
                        pt2=right_wrist_keypoint,
                        color=color,
                        thickness=self.thickness,
                        lineType=cv2.LINE_AA )

        return scene, hands_up_flag