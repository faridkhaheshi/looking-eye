from ..object_detection import ObjectDetector
from ..object_tracking import ObjectTracker


class VideoObjectAnalyzer:
    def __init__(self,
                 detector='ssd_mobilenet_v2',
                 detect_every_n_secs=10,
                 log=True):
        self.detector = ObjectDetector(detector, log=log)
        self.tracker = ObjectTracker()
        self.detection_frames = detect_every_n_secs

    def process_frame(self, frame, frame_count=0):
        if frame_count % self.detection_frames == 0:
            output_dict = self.detector.detect_people(frame)
            self.tracker.start_tracking(output_dict, frame)
        else:
            self.tracker.update(frame)
        return self.tracker.targets
