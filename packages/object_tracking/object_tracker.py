import dlib
import numpy as np


class ObjectTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.targets = {}
        self.trackers = []
        self.category_index = {}

    def update_category_index(self, category_index):
        self.category_index = category_index

    def add_tracker(self, box, image):
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(*box)
        tracker.start_track(image, rect)
        self.trackers.append(tracker)

    def update(self, image):
        for i in range(len(self.trackers)):
            confidence = self.trackers[i].update(image)
            pos = self.trackers[i].get_position()
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            # self.targets["scores"][i] = confidence
            self.targets["boxes"][i] = np.array(
                [start_x, start_y, end_x, end_y])
        return self.targets

    def start_tracking(self, detection_output_dict, image):
        self.reset()
        self.update_category_index(detection_output_dict["category_index"])
        self.targets = detection_output_dict
        for box in detection_output_dict["boxes"]:
            self.add_tracker(box, image)
