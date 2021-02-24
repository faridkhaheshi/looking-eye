import numpy as np
from cv2 import cv2
from ..utils.load_model import load_model


class ObjectDetector:
    def __init__(self, model_name="ssd_mobilenet_v2", log=True):
        model, model_info = load_model(model_name, log)
        self.model = model
        self.input_size = model_info["input_size"]
        self.categories = model_info["category_index"]

    def detect(self,
               image,
               nms=True,
               min_confidence=0.2,
               nms_threshold=0.4,
               look_for=None):
        """
        This methods performs the detection and returns the detected objects.
        Args:
          image: a numpy array representation of the image.
                The image should be in RGB and in this shape: (h, w, c).
          nms: a boolean indicating whethe Non-Max Suppression must be applied or not.
          min_confidence: the minimum confidence required for each detection as a float number
                between 0.0 and 1.0. Only detections whose confidence score is higher than 
                this value will be returned.
          nms_threshold: the threshold used for NMS algorithm. A float number between 0.0 and 1.0.
          look_for: a list of classes to restrict returned detections. e.g. ["person", "chair"].
                Only detections of the given classes will be returned.
        Returns:
          A dictionary with the following keys:
            {
              "boxes": a numpy array of shape (n, 4). n is the number of detected objects. 
                      for each detection (each row), the bounding box is reported in this form:
                        (x_start, y_start, x_end, y_end)
              "scores": a numpy array of shape (n,) containing confidence scores for each detection.
              "classes": a numpy array of shape (n,) containind class index for eaech detection.
              "category_index": a dictionary containing all classes in the following form
                      {
                        1: {'id': 1, 'name': 'person'}
                        2: {'id': 2, 'name': 'bicycle'},
                        ...
                        10: {id: 10, name: 'traffic light'},
                        ...
                      }
            }
        """

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image=image, size=self.input_size, crop=False, swapRB=True)
        self.model.setInput(blob)
        detections = self.model.forward()
        boxes_rel = detections[0, 0, :, 3:]
        confidences = detections[0, 0, :, 2]
        classes = detections[0, 0, :, 1].astype(np.int64)
        boxes_abs = (boxes_rel * np.array([w, h, w, h])).astype("int")

        if nms:
            boxes_list = boxes_abs.tolist()
            confidence_list = confidences.tolist()
            indices = cv2.dnn.NMSBoxes(
                boxes_list, confidence_list, min_confidence, nms_threshold)
            if len(indices):
                indices = indices[:, 0]
            else:
                indices = []
        else:
            indices = [i for (i, confidence) in enumerate(
                confidences) if confidence >= min_confidence]

        if look_for is not None:
            indices = [ind for ind in indices
                       if self.categories[classes[ind]]["name"] in look_for]

        boxes_abs = boxes_abs[indices]
        confidences = confidences[indices]
        classes = classes[indices]

        output_dict = {
            "boxes": boxes_abs,
            "scores": confidences,
            "classes": classes,
            "category_index": self.categories
        }
        return output_dict

    def detect_people(self, image, **args):
        return self.detect(image, **args, look_for=["person"])
