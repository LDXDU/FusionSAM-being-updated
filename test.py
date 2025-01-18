import os

import matplotlib.pyplot as plt
from prettytable import PrettyTable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import torch
from scipy.io import loadmat
import tqdm
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import data_loader, eva
from torch import optim, nn
from torch.utils import data
from torch.nn import functional as F

if __name__ == '__main__':
    model_type = 'vit_h'
    checkpoint = './ckpt/weights.pth'
    device = 'cuda'
    EVA = eva.Evaluator(15)

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)

    sam_model.eval()
    path_in = './SAM/data_emb'
    path_out = './result_feature'
    os.makedirs(path_out, exist_ok=True)
    test_files = data_loader.get_data(path_in, True)

    dataset_val = data_loader.DS(test_files, 'vis')

    val_loader = data.DataLoader(dataset_val, batch_size=1,
                                 shuffle=True, num_workers=4)

    for i, (datas) in enumerate(tqdm.tqdm(val_loader)):
        inputs_img = datas['img'].to(device)
        inputs_vis_emb = datas['vis_emb'].to(device)
        inputs_inf_emb = datas['inf_emb'].to(device)
        labels = datas['label'].to(device)
        cond = datas['cond'].to(device)
        path = datas['path'][0]
        data_in = loadmat(path)
        ori_img = data_in['fusion']
        ori_label = data_in['mask']
        input_image = sam_model.preprocess(inputs_img)

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            image_embedding = sam_model.adapter_model(inputs_vis_emb, inputs_inf_emb, image_embedding)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        image_embedding = image_embedding.cpu().numpy()[0]
        plt.figure(figsize=(12, 12))

        for j in range(len(image_embedding)):
            plt.subplot(16, 16, j + 1)
            plt.imshow(image_embedding[j])
            plt.axis('off')
        plt.savefig(path_out + '/' + path.split('/')[-1].split('.')[0] + '.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()
        masks = F.interpolate(
            low_res_masks,
            (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = torch.softmax(masks, dim=1)[0].cpu().numpy()
        masks = np.argmax(masks, axis=0).astype(np.uint8)

        masks = cv2.resize(masks, ori_label.shape[::-1], cv2.INTER_NEAREST)
        EVA.add_batch(ori_label, masks)

    table = PrettyTable(['', 'Acc', 'IoU', 'Recall', 'Precision', 'F1', 'Dice'])

    for i in range(len(EVA.Dice())):
        table.add_row([data_loader.label_name[i],
                       str(EVA.Pixel_Accuracy_Class()[i]),
                       str(EVA.Intersection_over_Union()[i]),
                       str(EVA.Recall()[i]),
                       str(EVA.Precision()[i]),
                       str(EVA.F1()[i]),
                       str(EVA.Dice()[i])
                       ])
    table.add_row(['mean',
                   str(EVA.Pixel_Accuracy_Class().mean()),
                   str(EVA.Intersection_over_Union().mean()),
                   str(EVA.Recall().mean()),
                   str(EVA.Precision().mean()),
                   str(EVA.F1().mean()),
                   str(EVA.Dice().mean())
                   ])
    print(table)
