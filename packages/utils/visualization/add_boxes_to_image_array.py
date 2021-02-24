import numpy as np
from object_detection.utils import visualization_utils as vis_util


def add_detected_boxes_to_image_array(image_array,
                                      detection_output_dict,
                                      line_thickness=4,
                                      min_score_thresh=0.0,
                                      swap_x_y=True):
    if swap_x_y:
        boxes = np.copy(detection_output_dict["boxes"])
        boxes[:, [0, 1]] = boxes[:, [1, 0]]
        boxes[:, [2, 3]] = boxes[:, [3, 2]]
    else:
        boxes = detection_output_dict["boxes"]
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_array,
        boxes,
        detection_output_dict["classes"],
        detection_output_dict["scores"],
        detection_output_dict["category_index"],
        line_thickness=line_thickness,
        min_score_thresh=min_score_thresh
    )
