import os
import wandb
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import json

class ImageLogger(Callback):
    def __init__(self, config, batch_frequency=1, logger_frequency=1000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, 
                 log_images_kwargs=None,specific_image_indices=None):
        super().__init__()
        self.config = config
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.logger_frequency = logger_frequency
        self.specific_image_indices = specific_image_indices
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.fixed_indices = [1,100,1000,1020]
        self.open_image = self.get_open_images_from_annotation(config['base_dir'], config['annotation_file'])
    def get_open_images_from_annotation(self, base_path, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        sorted_data = sorted(data, key=lambda x: x.get('source_image', ''))
        # 처음 8개의 고유한 source_image 경로를 추출
        unique_source_images = []
        for item in sorted_data:
            if 'source_image' in item and item['source_image'] not in unique_source_images:
                unique_source_images.append( base_path + item['source_image'])
                if len(unique_source_images) == 8:
                    break
        
        return unique_source_images
    def combine_images(self, images):
        # 모든 이미지를 하나의 큰 그리드로 결합
        all_grids = []
        for k in images:
            if k == "conditioning":
                continue
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            all_grids.append(grid)
        
        combined = torchvision.utils.make_grid(all_grids, nrow=1)
        combined = combined.transpose(0, 1).transpose(1, 2).squeeze(-1)
        combined = combined.numpy()
        combined = (combined * 255).astype(np.uint8)
        return combined

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, image_paths):
        base_path = os.path.join(save_dir, 'image_log', self.config['wandb_name'], split)
        
        for idx, image_path in enumerate(image_paths):
            # 이미지 파일 이름 추출
            image_filename = os.path.basename(image_path)
            
            # 각 이미지에 대한 고유 폴더 생성
            unique_folder = f"epoch-{current_epoch:06}_{image_filename}"
            folder_path = os.path.join(base_path, unique_folder)
            os.makedirs(folder_path, exist_ok=True)
            
            for k in images:
                if k == "conditioning":
                    continue
                
                # 현재 이미지만 선택
                img = images[k][idx].unsqueeze(0)
                
                grid = torchvision.utils.make_grid(img, nrow=1)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = f"{k}.png"
                path = os.path.join(folder_path, filename)
                Image.fromarray(grid).save(path)
            
            # 모든 이미지를 하나의 그리드로 결합
            combined_grid = self.combine_images({k: images[k][idx].unsqueeze(0) for k in images if k != "conditioning"})
            combined_path = os.path.join(folder_path, "combined.png")
            Image.fromarray(combined_grid).save(combined_path)
            wandb.log({f"{split}/{unique_folder}/combined": wandb.Image(combined_grid)}, step=global_step)

    
    def get_fixed_batch(self, pl_module):
        # 데이터셋에서 고정된 인덱스의 이미지 가져오기
        dataset = pl_module.trainer.train_dataloader.dataset
        fixed_batch = [dataset[i] for i in self.fixed_indices]
        # 배치로 변환 (실제 구현은 데이터셋 구조에 따라 달라질 수 있음)
        return self.collate_fn(fixed_batch)
    
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # print("log_img")
        if not self.disabled:
            # open_image 리스트에 있는 이미지만 선택
            selected_indices = [i for i, path in enumerate(batch['hint_path']) if path in self.open_image]
            if not selected_indices:
                return
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                # 고정된 인덱스의 이미지 가져오기
                selected_batch = {k: v[selected_indices] if isinstance(v, torch.Tensor) else [v[i] for i in selected_indices] for k, v in batch.items()}
                # Ensure intrinsics are included
                if 'source_camera_intrinsic' in batch:
                    selected_batch['source_camera_intrinsic'] = batch['source_camera_intrinsic'][selected_indices]
                if 'target_camera_intrinsic' in batch:
                    selected_batch['target_camera_intrinsic'] = batch['target_camera_intrinsic'][selected_indices]
                
                    images = pl_module.log_images(selected_batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            # 선택된 이미지 경로 전달
            selected_image_paths = [batch['hint_path'][i] for i in selected_indices]
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx,selected_image_paths)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % 1 == 0
        # return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
