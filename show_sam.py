import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
import tqdm
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import data_loader
from torch import optim, nn
from torch.utils import data
from torch.nn import functional as F


colors = list(data_loader.label_color.values())

if __name__ == '__main__':
    model_type = 'vit_h'
    checkpoint = './ckpt/weights.pth'
    device = 'cuda'

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)

    sam_model.eval()
    path_in = '/media/zyj/data_all/zyj/SAM/data_emb'
    test_files = data_loader.get_data(path_in, False)

    dataset_val = data_loader.DS(test_files)

    val_loader = data.DataLoader(dataset_val, batch_size=1,
                                 shuffle=True, num_workers=4)

    for i, (datas) in enumerate(tqdm.tqdm(val_loader)):
        inputs_img = datas['img'].to(device)
        labels = datas['label'].to(device)
        inputs_vis_emb = datas['vis_emb'].to(device)
        inputs_inf_emb = datas['inf_emb'].to(device)
        cond = datas['cond'].to(device)
        path = datas['path'][0]
        data_in = loadmat(path)
        ori_img = data_in['fusion']
        ori_label = data_in['mask']
        input_image = sam_model.preprocess(inputs_img)

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            image_embedding = sam_model.adapter_model(inputs_vis_emb, inputs_inf_emb, image_embedding)
            print(image_embedding.shape)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=cond,
            )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        masks = F.interpolate(
            low_res_masks,
            (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = torch.softmax(masks, dim=1)[0].cpu().numpy()
        masks = np.argmax(masks, axis=0).astype(np.uint8)
        masks = cv2.resize(masks, ori_label.shape[::-1], cv2.INTER_NEAREST)

        seg_masks = np.reshape(np.array(colors, np.uint8)[np.reshape(masks, [-1])], [*ori_label.shape, -1])
        ori_label = np.reshape(np.array(colors, np.uint8)[np.reshape(ori_label, [-1])], [*ori_label.shape, -1])
        # masks = colors[masks.flatten()]

        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.imshow(ori_img)

        plt.subplot(222)
        plt.imshow(ori_label)

        plt.subplot(223)
        plt.imshow(seg_masks)

        plt.subplot(224)
        plt.imshow(image_embedding.cpu().numpy()[0, 0])
        plt.show()
        break

