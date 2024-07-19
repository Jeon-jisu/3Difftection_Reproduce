import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class GeometryDataset(Dataset):
    def __init__(self, annotation_file, base_dir):
        self.base_dir = base_dir
        self.data = []
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Process annotations
        for annotation in annotations:
            source_image = os.path.join(base_dir, annotation['source_image'])
            target_image = os.path.join(base_dir, annotation['target_image'])
            
            self.data.append({
                'hint': source_image,
                'jpg': target_image,
                'source_camera_pose': annotation['source_camera_pose'],
                'target_camera_pose': annotation['target_camera_pose'],
                'source_camera_intrinsic': annotation['source_camera_intrinsic'],
                'target_camera_intrinsic': annotation['target_camera_intrinsic'],
                'txt':''
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load images
        source = cv2.imread(item['hint'])
        target = cv2.imread(item['jpg'])

        # Convert BGR to RGB.  OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # TODO : 정규화 범위 different 이슈 일단 둘다 -1~1로 변경
        # Normalize source images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # Convert camera poses and intrinsics to numpy arrays
        source_camera_pose = np.array(item['source_camera_pose'])
        target_camera_pose = np.array(item['target_camera_pose'])
        source_camera_intrinsic = np.array(item['source_camera_intrinsic'])
        target_camera_intrinsic = np.array(item['target_camera_intrinsic'])

        return {
            'hint': source,
            'jpg': target,
            'source_camera_pose': source_camera_pose,
            'target_camera_pose': target_camera_pose,
            'source_camera_intrinsic': source_camera_intrinsic,
            'target_camera_intrinsic': target_camera_intrinsic,
            'txt':''
        }