import os

import tqdm

from fusion.utils.tools import *
import cv2
import numpy as np
from scipy.io import loadmat, savemat
from fusion.nets.fusion_model import SwinFusion
import torch
from PIL import Image


def define_model():
    height = 128
    width = 128
    window_size = 8
    netG = SwinFusion(upscale=1,
                      img_size=(height, width),
                      window_size=window_size,
                      img_range=1.,
                      depths=[6, 6, 6, 6],
                      embed_dim=60,
                      num_heads=[6, 6, 6, 6],
                      mlp_ratio=2,
                      upsampler='',
                      resi_connection='1conv').to(device)
    key = netG.load_state_dict(
        torch.load('./fusion/checkpoint_mat_rgb/700_E.pth'))
    print(key)
    netG.eval()
    return netG


if __name__ == '__main__':
    device = 'cuda'
    path_label_in = './SAM/FMB/train/Label'

    mode = 'train'
    path_out = './SAM/data_emb/' + mode

    model = define_model()
    labels = []
    for file in tqdm.tqdm(os.listdir(path_out)):
        datas = loadmat(path_out + '/' + file)
        out = datas
        vis = datas['vis']
        inf = datas['inf']
        vis_emb = datas['vis_emb']
        inf_emb = datas['inf_emb']
        name = file.split('.')[0]
        label = np.array(Image.open(path_label_in + '/' + name + '.png').resize((256, 256), resample=Image.NEAREST))
        A_lab = cv2.cvtColor(vis, cv2.COLOR_RGB2Lab)
        with torch.no_grad():
            vis_emb = torch.from_numpy(vis_emb).type(torch.FloatTensor).to(device)[None]
            inf_emb = torch.from_numpy(inf_emb).type(torch.FloatTensor).to(device)[None]
            output = model(vis_emb, inf_emb)
            output = output.detach()[0].float().cpu()
        output = tensor2uint(output)[..., None]
        output = np.concatenate([output, A_lab[..., 1:]], axis=-1)
        output = cv2.cvtColor(output, cv2.COLOR_Lab2RGB)

        out['fusion'] = output
        out['mask'] = label
        assert output.shape[:2] == label.shape
        labels.extend(np.unique(label))
        savemat(path_out + '/' + file, out)
    print(np.unique(labels))
