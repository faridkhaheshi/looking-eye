import numpy as np
import PIL
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from .default_colors import DEFAULT_COLORS


def add_box_to_image_pil(image_pil, box, texts=[], thickness=4, color='red'):
    """
      This method adds a bounding box with the given texts to an image.
      Bounding box coordinates should be specified in absolute.

      Args:
        image: a PIL.Image object.
        box: coordinates of the box in this form: (x_min, y_min, x_max, y_max)
          which is the same as (left, top, right, bottom)
        color: color of the bounding box and the text container above (or below) it.
        thickness: line thickness as a number. Default: 4.
        texts: a list of texts to be shown for the bounding box. Each item of the list
          will be shown on a separate line.

      Returns nothing. Mutates the input image.
    """

    draw = ImageDraw.Draw(image_pil)
    (left, top, right, bottom) = box
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)

    if len(texts) == 0:
        return

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    text_heights = [font.getsize(t)[1] for t in texts]
    total_texts_height = (1 + 2 * 0.05) * sum(text_heights)

    if top > total_texts_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_texts_height

    for text in texts[::-1]:
        text_width, text_height = font.getsize(text)
        margin = np.ceil(0.05 * text_height)

        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  text,
                  fill='black',
                  font=font)
        text_bottom -= text_height + 2 * margin


def add_box_to_image_array(image_array, box, **args):
    """
      This method is the same as add_box_to_image but accepts numpy array as inputs.
      All other arguments will be passed to the add_box_to_image method.
    """

    image_pil = Image.fromarray(np.uint8(image_array)).convert('RGB')
    add_box_to_image_pil(image_pil, box, **args)
    np.copyto(image_array, np.array(image_pil))


def add_box_to_image(image, box, **args):
    if type(image) == np.ndarray:
        add_box_to_image_array(image, box, **args)
    elif type(image) == PIL.Image.Image:
        add_box_to_image_pil(image, box, **args)


def add_boxes_to_image(image, boxes, colors=DEFAULT_COLORS, texts_list=(), thickness=4):
    """
      This method adds all bounding boxes to the image.

      Args:
        - image: the RGB image as PIL.Image or Numpy array.
        - boxes: a numpy array in shape (N, 4) where is the number of boxes.
          each box is presented as (x_min, y_min, x_max, y_max).
        - colors: a list of colors to be used. Each color in the list corresponds to 
          one box.
        - texts_list: for Each box, a list of texts to be shown with the box. Each text will be 
          shown on a separate line.
        - thickness: line thickness as a number. Default: 4.

      Returns nothing. Mutates the given image.
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    all_colors = len(colors)
    for i in range(boxes_shape[0]):
        texts = ()
        if texts_list:
            texts = texts_list[i]
        add_box_to_image(image,
                         boxes[i, :],
                         color=colors[i % all_colors],
                         texts=texts,
                         thickness=thickness)
