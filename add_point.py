import os

import matplotlib.pyplot as plt
import tqdm

from fusion.utils.tools import *
import cv2
import numpy as np
from scipy.io import loadmat, savemat


def get_point(mask):
    cls = np.unique(mask)
    out = np.zeros_like(mask)
    for c in cls:
        if c == 0:
            continue
        mask_ = mask == c
        x, y = np.where(mask_ > 0)
        length = len(x)
        index = np.random.choice(np.arange(0, length), 10)
        out[x[index], y[index]] = c
    return out


if __name__ == '__main__':
    path_in = './SAM/data_emb'
    for file in os.listdir(path_in):
        path_in1 = path_in + '/' + file
        for mat_file in tqdm.tqdm(os.listdir(path_in1)):
            data = loadmat(path_in1 + '/' + mat_file)
            cond = get_point(data['mask'])
            data['cond'] = cond
            savemat(path_in1 + '/' + mat_file, data)
