import dlib
import numpy as np


class TrackedPerson:
    next_person_id = 0

    def __init__(self, frame, box, score=-1.0):
        self.tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(*box)
        self.tracker.start_track(frame, rect)
        self.detection_score = score
        self.confidence = 11.0
        self.time_since_update = 0
        self.id = TrackedPerson.next_person_id
        TrackedPerson.next_person_id += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def get_box(self):
        pos = self.tracker.get_position()
        start_x = int(pos.left())
        start_y = int(pos.top())
        end_x = int(pos.right())
        end_y = int(pos.bottom())

        return np.array([start_x, start_y, end_x, end_y])

    def predict(self, frame):
        self.confidence = self.tracker.update(frame)

        self.age += 1
        # if (self.time_since_update > 0):
        #     self.hit_streak = 0
        self.time_since_update += 1
        return self.get_box()

    def get_labels(self):
        confidence = "conf: {:.2f}".format(self.confidence)
        id_label = "ID: {}".format(self.id)
        detection_score = "score: {:.2f}%".format(100*self.detection_score)
        return [id_label, confidence, detection_score]
