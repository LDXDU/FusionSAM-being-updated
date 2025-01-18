import glob
import PIL.Image
import torch
import numpy as np
from torch.utils import data


def get_path(path_all):
    files = glob.glob(path_all + '/*.png')
    return files


def preprocess_input(img):
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-3)
    return img * 2 - 1


class DS(data.Dataset):
    def __init__(self, path):
        super(DS, self).__init__()
        self.path = path

    def __getitem__(self, item):
        x_img = PIL.Image.open(self.path[item]).convert('RGB').resize((256, 256))
        x_img = preprocess_input(np.array(x_img)).transpose(2, 0, 1)

        x_tensor = torch.FloatTensor(x_img)
        return {'data': x_tensor}

    def __len__(self):
        return len(self.path)

