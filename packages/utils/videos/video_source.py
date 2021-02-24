from cv2 import cv2


class VideoSource:
    def __init__(self, source=0):
        self.video_capture = cv2.VideoCapture(source)
        self.stream_enabled = True

    def release(self):
        self.video_capture.release()

    def reset_frame_pointer(self):
        self.set_frame_pointer(0)

    def get_length_in_sec(self):
        return self.get_total_frames()/self.get_fps()

    def get_fps(self):
        return self.video_capture.get(cv2.CAP_PROP_FPS)

    def get_total_frames(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_pointer(self):
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))

    def set_frame_pointer(self, frame_number):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_frame_at_sec(self, frame_second, **args):
        frame_number = int(frame_second * self.get_fps())
        return self.get_frame(frame_number=frame_number, **args)

    def get_frame(self, frame_number=None, bgr=False):
        if frame_number is not None:
            self.set_frame_pointer(frame_number)

        ret, frame = self.video_capture.read()
        if frame is None or not ret:
            return None

        if not bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def stop_stream(self):
        self.stream_enabled = False

    def enable_stream(self): self.stream_enabled = True

    def stream(self):
        self.enable_stream()
        frame_count = 0
        while self.stream_enabled:
            frame = self.get_frame()
            if frame is None:
                break
            yield frame, frame_count
            frame_count += 1
