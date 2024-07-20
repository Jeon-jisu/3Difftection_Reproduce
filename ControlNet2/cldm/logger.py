import os
import wandb
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class ImageLogger(Callback):
    def __init__(self, config, batch_frequency=4000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.config = config
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        base_path = os.path.join(save_dir, 'image_log', self.config['wandb_name'], split)
        
        # 각 이미지 세트에 대한 고유 폴더 생성
        unique_folder = f"gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}"
        folder_path = os.path.join(base_path, unique_folder)
        os.makedirs(folder_path, exist_ok=True)
        
        for k in images:
            if k == "conditioning":
                continue
            
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            # 이부분이 정규화 맞춰주는 부분
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}.png"
            path = os.path.join(folder_path, filename)
            Image.fromarray(grid).save(path)
            # WandB에 이미지 로깅
            wandb.log({f"{split}_{k}": wandb.Image(grid)}, step=global_step)
        # 모든 이미지를 하나의 그리드로 결합
        combined_grid = self.combine_images(images)
        combined_path = os.path.join(folder_path, "combined.png")
        Image.fromarray(combined_grid).save(combined_path)
        wandb.log({f"{split}/{unique_folder}/combined": wandb.Image(combined_grid)}, step=global_step)

            
    @rank_zero_only
    def log_fixed_images(self, pl_module, batch, current_epoch):
        if self.fixed_images is None:
            self.fixed_images = batch[:min(4, len(batch))]

        with torch.no_grad():
            images = pl_module.log_images(self.fixed_images, split="fixed", **self.log_images_kwargs)

        for k in images:
            if k == "conditioning":
                continue
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"fixed_{k}_epoch-{current_epoch:04d}.png"
            path = os.path.join(pl_module.logger.save_dir, "image_log", "fixed", filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            wandb.log({f"fixed_{k}": wandb.Image(grid)}, step=current_epoch)
            
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
