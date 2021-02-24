from PIL import Image


def save_image(image_array, file_name):
    im = Image.fromarray(image_array)
    im.save(file_name)
