from cv2 import cv2
from object_detection.utils import label_map_util
import os.path

SUPPORTED_MODEL = {
    "ssd_inception_v2": {
        "dir": "ssd_inception_v2_coco_2017_11_17",
        "starting_index": 1,
        "size": (300, 300)
    },
    "ssd_mobilenet_v1": {
        "dir": "ssd_mobilenet_v1_coco_2017_11_17",
        "starting_index": 1,
        "size": (300, 300)
    },
    "ssd_mobilenet_v2": {
        "dir": "ssd_mobilenet_v2_coco_2018_03_29",
        "starting_index": 1,
        "size": (300, 300)
    },
    "faster_rcnn_inception_v2": {
        "dir": "faster_rcnn_inception_v2_coco_2018_01_28",
        "size": (600, 600),
        "starting_index": 0
    },
    "faster_rcnn_resnet50": {
        "dir": "faster_rcnn_resnet50_coco_2018_01_28",
        "size": (600, 600),
        "starting_index": 0
    }
}

file_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(file_path, "./../../dnn_models")


def load_model(model_name, log=True):
    if not model_name in SUPPORTED_MODEL:
        return None, None

    model_folder = "{}/{}/".format(base_path,
                                   SUPPORTED_MODEL[model_name]["dir"])
    pb_file = model_folder + "frozen_inference_graph.pb"
    pbtxt_file = model_folder + 'graph.pbtxt'
    labels_file = "{}/mscoco_label_map.pbtxt".format(base_path)
    label_starting_index = SUPPORTED_MODEL[model_name]["starting_index"]
    network_input_size = SUPPORTED_MODEL[model_name]["size"]
    category_index = label_map_util.create_category_index_from_labelmap(
        labels_file, use_display_name=True)
    if label_starting_index == 0:
        category_index = {(key - 1): {"id": value["id"] - 1, "name": value["name"]}
                          for (key, value) in category_index.items()}
    if log:
        print("[INFO] loading model...")
    net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
    if log:
        print("Done.")
    net_info = {
        "input_size": network_input_size,
        "category_index": category_index
    }
    return net, net_info
