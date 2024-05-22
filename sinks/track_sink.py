import supervision as sv
from inference.core.interfaces.camera.entities import VideoFrame
from tools.video_info import VideoInfo


class TrackSink:
    def __init__(
        self,
        iou: float = 0.7,
    ) -> None:
        self.tracker = sv.ByteTrack(minimum_matching_threshold=iou)

    def update_tracks(self, detections, frame: VideoFrame) -> None:
        detections = self.tracker.update_with_detections(detections)
        
        return detections