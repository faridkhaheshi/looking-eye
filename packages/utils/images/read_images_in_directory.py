import pathlib
from .read_image import read_image


def read_images_in_directory(directory_path, regex="*.jpg"):
    dir_path = pathlib.Path(directory_path)
    images_paths = sorted(list(dir_path.glob(regex)))
    return [read_image(image_path) for image_path in images_paths]
