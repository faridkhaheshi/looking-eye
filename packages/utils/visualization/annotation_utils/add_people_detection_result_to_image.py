from .add_box_to_image import add_boxes_to_image


def add_people_detection_result_to_image(image, output_dict, **args):
    boxes = output_dict["boxes"]
    scores = output_dict["scores"]
    texts = [["{:.2f}%".format(100.0*score)] for score in scores]
    add_boxes_to_image(image, boxes, texts_list=texts, **args)
