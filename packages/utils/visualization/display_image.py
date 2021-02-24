from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import display
from IPython import get_ipython
from cv2 import cv2


def display_image(image_array, bgr=False, figure=None):
    try:
        i_python = get_ipython().has_trait('kernel')
    except:
        i_python = False

    rgb = image_array
    if bgr:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    if i_python:
        print("running display")
        return display(Image.fromarray(rgb))
    elif figure is not None:
        handle = figure.imshow(rgb)
        plt.show()
        return handle
    else:
        plt.figure()
        plt.axis('off')
        handle = plt.imshow(rgb)
        plt.show()
        return handle
