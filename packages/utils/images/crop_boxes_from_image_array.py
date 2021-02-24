def crop_boxes_from_image_array(boxes, image_array):
    crops = []
    for box in boxes:
        x_min = max(0, box[0])
        y_min = max(0, box[1])
        x_max = min(box[2], image_array.shape[1])
        y_max = min(box[3], image_array.shape[0])
        crops.append(image_array[y_min:y_max, x_min: x_max, :])
    return crops
