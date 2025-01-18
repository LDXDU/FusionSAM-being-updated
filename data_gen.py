import os

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from vqgan.vq_gan.model import VQGAN
from vqgan.vq_gan.config import config
from data_loader import *

if __name__ == '__main__':
    cfg = config()
    device = 'cuda'
    net_vis = VQGAN(cfg).load_from_checkpoint(
        './vqgan/checkpoint_Visible/lightning_logs/version_0/checkpoints/latest_checkpoint-v2.ckpt').to(device)
    net_vis.eval()

    net_inf = VQGAN(cfg).load_from_checkpoint(
        './vqgan/checkpoint_Infrared/lightning_logs/version_0/checkpoints/latest_checkpoint-v2.ckpt').to(device)
    net_inf.eval()

    path_Infrared_in = './SAM/FMB/test/Infrared'
    path_Visible_in = './SAM/FMB/test/Visible'

    mode = 'test'
    path_out = './SAM/data_emb/' + mode
    os.makedirs(path_out, exist_ok=True)

    for file in tqdm.tqdm(os.listdir(path_Infrared_in)):
        inf_img_save = PIL.Image.open(path_Infrared_in + '/' + file).convert('RGB').resize((256, 256))
        vis_img_save = PIL.Image.open(path_Visible_in + '/' + file).convert('RGB').resize((256, 256))

        inf_img = preprocess_input(np.array(inf_img_save)).transpose(2, 0, 1)
        vis_img = preprocess_input(np.array(vis_img_save)).transpose(2, 0, 1)

        with torch.no_grad():
            x_tensor = torch.FloatTensor(inf_img)[None].to(device)
            y = net_inf.encode(
                x_tensor, quantize=False, include_embeddings=True)
            # normalize to -1 and 1
            inf_emb = ((y - net_inf.codebook.embeddings.min()) /
                       (net_inf.codebook.embeddings.max() -
                        net_inf.codebook.embeddings.min())) * 2.0 - 1.0
            inf_emb = inf_emb[0]

            x_tensor = torch.FloatTensor(vis_img)[None].to(device)
            y = net_vis.encode(
                x_tensor, quantize=False, include_embeddings=True)
            # normalize to -1 and 1
            vis_emb = ((y - net_vis.codebook.embeddings.min()) /
                       (net_vis.codebook.embeddings.max() -
                        net_vis.codebook.embeddings.min())) * 2.0 - 1.0
            vis_emb = vis_emb[0]

        out = {
            'vis': np.array(vis_img_save),
            'inf': np.array(inf_img_save),
            'vis_emb': vis_emb.cpu().numpy(),
            'inf_emb': inf_emb.cpu().numpy(),
        }
        savemat(path_out + '/' + file.split('.')[0] + '.mat', out)
