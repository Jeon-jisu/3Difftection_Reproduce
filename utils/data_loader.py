from torch.utils.data import Dataset
import torch
import os
import json
import cv2
import numpy as np


class Omni3DDataset(Dataset):
    def __init__(self, data_dir, dataset_type, split="train", resolution=256):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.split = split
        self.resolution = resolution

        if dataset_type == "ARKitScenes":
            self.annotation_file = os.path.join(data_dir, f"{split}_annotations.json")
        elif dataset_type == "SUN-RGBD":
            self.annotation_file = os.path.join(data_dir, f"{split}_annotations.json")
        elif dataset_type == "Omni-Indoor":
            self.annotation_file = os.path.join(data_dir, f"{split}_annotations.json")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        with open(self.annotation_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.data_dir, ann["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution, self.resolution))

        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Process 3D bounding boxes
        boxes_3d = torch.tensor(ann["3d_boxes"])

        # Process camera pose if available (for ARKitScenes)
        camera_pose = torch.tensor(
            ann.get("camera_pose", [0, 0, 0, 1, 0, 0, 0])
        )  # Default quaternion if not available

        return {
            "image": img,
            "boxes_3d": boxes_3d,
            "camera_pose": camera_pose,
        }
