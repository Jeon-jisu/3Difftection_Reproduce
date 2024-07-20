from torch.utils.data import Dataset
import torch
import os
import json
import cv2
import numpy as np


class Omni3DDataset(Dataset):
    def __init__(self, data_dir, resolution, num_control_channels=6, split="train", num_views=2, aggregation_method="attention", 
                 warp_last_n_stages=2, input_channels=3, output_channels=64, 
                 num_blocks=5, base_channels=32):
        self.data_dir = data_dir
        self.resolution = resolution
        self.split = split
        self.annotation_file = os.path.join(data_dir, f"{split}_annotations.json")
        
        with open(self.annotation_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution, self.resolution))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 소스 이미지 로드
        source_img_path = os.path.join(self.data_dir, ann["source_image"])
        source_image = self.load_image(source_img_path)
        
        # 타겟 이미지 로드
        target_img_path = os.path.join(self.data_dir, ann["target_image"])
        target_image = self.load_image(target_img_path)
        
        # 카메라 포즈 및 내부 파라미터 로드
        source_camera_pose = torch.tensor(ann["source_camera_pose"])
        target_camera_pose = torch.tensor(ann["target_camera_pose"])
        source_image_timestamp = torch.tensor(ann["source_image_timestamp"])
        target_image_timestamp = torch.tensor(ann["target_image_timestamp"])
        source_camera_intrinsic = torch.tensor(ann["source_camera_intrinsic"])
        target_camera_intrinsic = torch.tensor(ann["target_camera_intrinsic"])
        
        timestep = torch.abs(target_image_timestamp - source_image_timestamp)

        
        return {
            "source_image": source_image,
            "target_image": target_image,
            "source_camera_pose": source_camera_pose,
            "target_camera_pose": target_camera_pose,
            "source_image_timestamp":source_image_timestamp,
            "target_image_timestamp":target_image_timestamp,
            "source_camera_intrinsic": source_camera_intrinsic,
            "target_camera_intrinsic": target_camera_intrinsic,
            "timestep": timestep,
        }


        