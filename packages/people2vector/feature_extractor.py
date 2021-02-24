from .model import ft_net, ft_net_dense
from .transform_image import transform_images, transform_image
import torch.nn as nn
import torch
from torch.autograd import Variable

model_dict = {
    "ft_ResNet50": ft_net,
    "ft_net_dense": ft_net_dense
}


def fliplr(input_tensor):
    '''flip horizontal'''
    inv_idx = torch.arange(input_tensor.size(
        3)-1, -1, -1).long()  # N x C x H x W
    flipped = input_tensor.index_select(3, inv_idx)
    return flipped


class FeatureExtractor:
    def __init__(self,
                 saved_model_path='./person_reID_models/ft_ResNet50/net_last.pth',
                 model_name='ft_ResNet50',
                 class_num=751,
                 **args
                 ):
        model_class = model_dict.get(model_name, model_dict["ft_ResNet50"])
        self.model = model_class(class_num, **args)
        self.model.load_pretrained_state(saved_model_path)
        self.model.classifier.classifier = nn.Sequential()
        self.model.eval()

    def extract(self, input_tensor, mirror=True):
        with torch.no_grad():
            features = torch.FloatTensor()
            n, c, h, w = input_tensor.size()
            ff = torch.FloatTensor(n, 512).zero_()
            if mirror:
                for i in range(2):
                    if i == 1:
                        input_tensor = fliplr(input_tensor)
                    input_var = Variable(input_tensor)
                    outputs = self.model(input_var)
                    ff += outputs
            else:
                input_var = Variable(input_tensor)
                outputs = self.model(input_var)
                ff += outputs

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff.data.cpu()), 0)
            return features.numpy()

    def extract_images_features(self, images, is_array=False):
        input_tensor = transform_images(images, is_array=is_array)
        return self.extract(input_tensor)

    def extract_image_features(self, image, is_array=False):
        input_tensor = transform_image(image, is_array=is_array)
        return self.extract(input_tensor)
