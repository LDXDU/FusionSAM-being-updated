import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import optim, nn
from torch.utils import data
from torch.nn import functional as F
import torch
import tqdm
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import data_loader

if __name__ == '__main__':
    model_type = 'vit_h'
    checkpoint = './segment_anything/ckpt/sam_vit_h_4b8939.pth'
    device = 'cuda'
    batch_size = 2
    num_epochs = 100
    save_loss_min = 100
    weights_path = './ckpt_15_no_emb_v1'
    os.makedirs(weights_path, exist_ok=True)

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.train()

    path_in = './SAM/data_emb'

    train_files = data_loader.get_data(path_in, False)
    test_files = data_loader.get_data(path_in, True)
    train_files.extend(test_files)
    dataset_train = data_loader.DS(train_files)
    dataset_val = data_loader.DS(test_files)

    train_loader = data.DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    # train_loader = [d for dl in [train_loader, val_loader] for d in dl]
    optimizer = optim.Adam(sam_model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        dt_size = len(train_loader.dataset)
        dt_size_val = len(val_loader.dataset)
        epoch_loss = 0
        pbar = tqdm.tqdm(
            total=dt_size // batch_size,
            desc=f'Epoch {epoch + 1} / {num_epochs}',
            postfix=dict,
            miniters=.3
        )
        sam_model.train()
        for i, (datas) in enumerate(train_loader):
            inputs_img = datas['img'].to(device)
            inputs_vis_emb = datas['vis_emb'].to(device)
            inputs_inf_emb = datas['inf_emb'].to(device)
            labels = datas['label'].to(device)
            cond = datas['cond'].to(device)
            input_image = sam_model.preprocess(inputs_img)
            optimizer.zero_grad()
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
            # image_embedding = sam_model.adapter_model(inputs_vis_emb, inputs_inf_emb, image_embedding)

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
            loss = loss_fn(masks, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix(**{
                'train_loss': epoch_loss / (i + 1),
            })
            pbar.update(1)

        pbar.close()
        pbar = tqdm.tqdm(
            total=dt_size_val // batch_size,
            desc=f'Val_Epoch {epoch + 1} / {num_epochs}',
            postfix=dict,
            miniters=.3
        )
        epoch_loss_val = 0
        sam_model.eval()

        for i, (datas) in enumerate(val_loader):
            inputs_img = datas['img'].to(device)
            inputs_vis_emb = datas['vis_emb'].to(device)
            inputs_inf_emb = datas['inf_emb'].to(device)
            labels = datas['label'].to(device)
            cond = datas['cond'].to(device)
            input_image = sam_model.preprocess(inputs_img)
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
                # image_embedding = sam_model.adapter_model(inputs_vis_emb, inputs_inf_emb, image_embedding)

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
            loss = loss_fn(masks, labels)
            epoch_loss_val += loss.item()
            pbar.set_postfix(**{
                'val_loss': epoch_loss_val / (i + 1),
            })
            pbar.update(1)
        pbar.close()
        if save_loss_min > epoch_loss_val / i:
            save_loss_min = epoch_loss_val / i
            torch.save(sam_model.state_dict(), weights_path + '/weights.pth')
    print("训练完成！")
