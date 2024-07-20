import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from custom_dataset import GeometryDataset

base_dir = '/node_data/urp24s_jsjeon/3Difftection_Reproduce2/ControlNet/testraw/'
annotation_file = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/testraw/annotations.json'
dataset = GeometryDataset(annotation_file, base_dir)
print(f"Dataset size: {len(dataset)}")

# 임의의 인덱스로 아이템 확인 (여기서는 0번째 아이템을 사용)
item = dataset[0]
jpg = item['jpg']
hint = item['hint']

print(f"Target image shape: {jpg.shape}")
print(f"Source image shape: {hint.shape}")

# 추가 정보 출력
print(f"Source camera pose: {item['source_camera_pose']}")
print(f"Target camera pose: {item['target_camera_pose']}")
print(f"Source camera intrinsic:\n{item['source_camera_intrinsic']}")
print(f"Target camera intrinsic:\n{item['target_camera_intrinsic']}")

