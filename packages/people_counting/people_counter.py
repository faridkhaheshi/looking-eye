from ..object_detection import ObjectDetector
from ..object_tracking import ObjectTracker


class PeopleCounter:
    def __init__(self,
                 detector='ssd_mobilenet_v2',
                 detect_every_n_secs=10,
                 log=True
                 ):
        self.tracked_people = {}
        self.next_id = 0
        self.detector = ObjectDetector(detector, log=log)
        self.tracker = ObjectTracker()
        self.detection_frames = detect_every_n_secs
        self.frame_count = 0

    def process_frame(self, frame):
        if self.frame_count % self.detection_frames == 0:
            expected_targets = self.tracker.update(frame)
            detected_targets = self.detector.detect_people(frame)
            self.tracker.start_tracking(detected_targets, frame)
        else:
            self.tracker.update(frame)

        return self.tracker.targets
