import torch
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def transform_image(image, is_array=False):
    if is_array:
        image = Image.fromarray(image)
    image_t = transform(image)
    batch_t = torch.unsqueeze(image_t, 0)
    return batch_t


def transform_images(image_list, is_array=False):
    images_t_list = []
    for image in image_list:
        if is_array:
            image = Image.fromarray(image)
        image_t = transform(image)
        images_t_list.append(image_t)
    return torch.stack(images_t_list, dim=0)
