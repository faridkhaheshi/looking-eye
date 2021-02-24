from packages.utils import load_model
from packages.utils.visualization import display_image
from packages.utils.images import read_image
from packages.object_detection import ObjectDetector

from matplotlib import pyplot as plt

detector = ObjectDetector('ssd_inception_v2')

image = read_image('images/image2.jpg')
display_image(image)
