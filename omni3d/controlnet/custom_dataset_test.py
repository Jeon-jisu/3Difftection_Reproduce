import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from custom_dataset import GeometryDataset

def inspect_dataset_item(dataset, index):
    print(f"Inspecting item at index: {index}")
    
    # 데이터셋 크기 확인
    print(f"Dataset size: {len(dataset)}")

    # 지정된 인덱스의 아이템 가져오기
    item = dataset[index]
    
    # 이미지 정보 출력
    jpg = item['jpg']
    hint = item['hint']
    print(f"Target image shape: {jpg.shape}")
    print(f"Source image shape: {hint.shape}")

    # 추가 정보 출력
    print(f"Source camera pose: {item['source_camera_pose']}")
    print(f"Target camera pose: {item['target_camera_pose']}")
    print(f"Source camera intrinsic:\n{item['source_camera_intrinsic']}")
    print(f"Target camera intrinsic:\n{item['target_camera_intrinsic']}")
    
    # 이미지 경로 출력 (GeometryDataset이 이 정보를 제공한다고 가정)
    if 'hint_path' in item and 'jpg_path' in item:
        print(f"Source image path: {item['hint_path']}")
        print(f"Target image path: {item['jpg_path']}")
    else:
        print("Image paths are not available in the dataset item.")

# 함수 사용 예시
def main():
    base_dir = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/'
    annotation_file = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/annotations.json'
    dataset = GeometryDataset(annotation_file, base_dir)

    # 예: 1000번째 아이템 검사
    inspect_dataset_item(dataset, 1000)
    inspect_dataset_item(dataset, 2000)
    inspect_dataset_item(dataset, 3000)
    inspect_dataset_item(dataset, 4000)
    

    # 다른 인덱스로도 시도 가능
    # inspect_dataset_item(dataset, 0)
    # inspect_dataset_item(dataset, 500)
    # inspect_dataset_item(dataset, len(dataset) - 1)

if __name__ == "__main__":
    main()