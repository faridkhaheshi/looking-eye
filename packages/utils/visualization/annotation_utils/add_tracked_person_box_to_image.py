from .add_box_to_image import add_box_to_image
from .default_colors import DEFAULT_COLORS


def add_tracked_person_box_to_image(image, tracked_person):
    color = DEFAULT_COLORS[tracked_person.id % len(DEFAULT_COLORS)]
    add_box_to_image(image,
                     tracked_person.get_box(),
                     texts=tracked_person.get_labels(),
                     color=color)
