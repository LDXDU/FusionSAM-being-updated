import numpy as np
from matplotlib import pyplot as plt

from vqgan.vq_gan.model import VQGAN
from torch.utils.data import DataLoader
from scipy.io import loadmat
from vqgan.vq_gan.config import config
from data_loader import *

if __name__ == '__main__':
    cfg = config()
    device = 'cuda'
    net = VQGAN(cfg).load_from_checkpoint(
        './vqgan/checkpoint_Visible/lightning_logs/version_0/checkpoints/latest_checkpoint-v2.ckpt').to(device)
    net.eval()

    img_path = '/media/zyj/data_all/zyj/SAM/FMB/test/Visible/00088.png'

    img = x_img = PIL.Image.open(img_path).convert('RGB').resize((256, 256))
    x_img = preprocess_input(np.array(x_img)).transpose(2, 0, 1)

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_img)[None].to(device)
        y = net.encode(
            x_tensor, quantize=False, include_embeddings=True)
        # normalize to -1 and 1
        y = ((y - net.codebook.embeddings.min()) /
             (net.codebook.embeddings.max() -
              net.codebook.embeddings.min())) * 2.0 - 1.0

        x = (((y + 1.0) / 2.0) * (net.codebook.embeddings.max() -
                                  net.codebook.embeddings.min())) + net.codebook.embeddings.min()

        x = net.decode(x, quantize=True)

    x = x.cpu().numpy()[0].transpose(1, 2, 0)
    x = (x + 1) / 2
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(np.array(img))

    plt.subplot(122)
    plt.imshow(x)

    plt.show()
