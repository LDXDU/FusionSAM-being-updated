import os
import pytorch_lightning as pl
from vqgan.vq_gan.model import VQGAN
from pytorch_lightning.callbacks import ModelCheckpoint
import data_loader
from torch.utils.data import DataLoader
from vqgan.vq_gan.config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run():
    files = data_loader.get_path('./SAM/FMB/train/Infrared')
    train_dataloader = DataLoader(data_loader.DS(files),
                                  batch_size=8,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(data_loader.DS(files),
                                batch_size=8,
                                shuffle=False, num_workers=4)
    cfg = config()
    net = VQGAN(cfg)
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)
    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'

    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(net, train_dataloader, val_dataloader)


if __name__ == '__main__':
    device = 'cuda'

    run()
