import numpy as np
from PIL import Image


def read_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np
