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
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from pytorch_grad_cam.grad_cam import GradCAM
colors = list(data_loader.label_color.values())


class build(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.to(device)

        self.model.eval()

    def forward(self, inputs_img):
        input_image = self.model.preprocess(inputs_img)
        with torch.no_grad():
            image_embedding = self.model.image_encoder(input_image)
        image_embedding = self.model.adapter_model(inputs_vis_emb, inputs_inf_emb, image_embedding)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=cond,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        return low_res_masks


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


if __name__ == '__main__':
    model_type = 'vit_h'
    checkpoint = './ckpt/weights.pth'
    device = 'cuda'

    sam_model = build()

    path_in = '/media/zyj/data_all/zyj/SAM/data_emb'
    path_features = './result_cam'
    os.makedirs(path_features, exist_ok=True)
    test_files = data_loader.get_data(path_in, False)

    dataset_val = data_loader.DS(test_files)

    val_loader = data.DataLoader(dataset_val, batch_size=1,
                                 shuffle=True, num_workers=4)
    target_layers = [sam_model.model.adapter_model]

    round_category = 1
    for i, (datas) in enumerate(tqdm.tqdm(val_loader)):
        inputs_img = datas['img'].to(device)
        labels = datas['label'].to(device)
        inputs_vis_emb = datas['vis_emb'].to(device)
        inputs_inf_emb = datas['inf_emb'].to(device)
        cond = datas['cond'].to(device)
        path = datas['path'][0]
        data_in = loadmat(path)

        ori_img = data_in['vis'] / 255.
        ori_label = data_in['mask']

        with torch.no_grad():
            low_res_masks = sam_model(inputs_img)
        masks = torch.softmax(low_res_masks, dim=1)[0]

        round_mask = torch.argmax(masks, dim=0).detach().cpu().numpy()
        round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
        round_mask_float = np.float32(round_mask == round_category)
        targets = [SemanticSegmentationTarget(round_category, round_mask_float)]

        with GradCAM(model=sam_model, target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=inputs_img,
                                targets=targets)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, ori_img.shape[:2][::-1])
            cam_image = show_cam_on_image(ori_img, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(path_features + '/' + path.split('/')[-1].split('.')[0] + '.png')

        # plt.figure(figsize=(12, 12))
        # plt.imshow(cam_image)
        # plt.show()
        #
        # break
