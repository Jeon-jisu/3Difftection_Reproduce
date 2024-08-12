"""
이 스크립트는 다음과 같은 작업을 수행합니다:

1. 지정된 경로에서 비디오 데이터를 읽어 카메라 포즈, 내부 파라미터, 이미지 경로 등의 정보를 수집합니다.
2. 각 이미지 쌍에 대해 회전 각도를 계산합니다.
3. 수집된 정보를 JSON 형식의 어노테이션 파일로 저장합니다.
4. 저장된 어노테이션 파일을 읽어 회전 각도가 20도 이하인 항목만 필터링합니다.
5. 필터링된 데이터를 새로운 JSON 파일로 저장합니다.

이 스크립트는 3D 이미지 쌍 데이터셋을 생성하고 필터링하는 데 사용됩니다.
필터링은 큰 회전 각도를 가진 이미지 쌍을 제거하여 데이터의 품질을 향상시키는 데 도움을 줍니다.

사용법:
1. base_path 변수를 원하는 데이터 경로로 설정합니다.
2. 스크립트를 실행합니다.
3. 결과로 두 개의 JSON 파일이 생성됩니다:
   - annotations_not_rotate2.json: 모든 어노테이션 데이터
   - annotations_not_rotate_filter.json: 회전 각도가 20도 이하인 항목만 포함된 필터링된 데이터
"""

import os
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

def round_to_3_decimal(value):
    return round(float(value), 3)

def process_traj_file(traj_file):
    pose_data = {}
    with open(traj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            timestamp = round_to_3_decimal(float(parts[0]))
            pose_data[timestamp] = {
                'rotation': [float(parts[1]), float(parts[2]), float(parts[3])],
                'translation': [float(parts[4]), float(parts[5]), float(parts[6])]
            }
    return pose_data

def process_pincam_file(pincam_file):
    with open(pincam_file, 'r') as f:
        data = f.read().strip().split()
    return {
        'width': float(data[0]),
        'height': float(data[1]),
        'focal_length_x': float(data[2]),
        'focal_length_y': float(data[3]),
        'principal_point_x': float(data[4]),
        'principal_point_y': float(data[5])
    }

def find_closest_timestamp(timestamp, pose_data, max_diff=0.1):
    close_timestamps = [ts for ts in pose_data.keys() if abs(ts - timestamp) <= max_diff]
    if not close_timestamps:
        return None
    closest = min(close_timestamps, key=lambda x: abs(x - timestamp))
    return closest

def calculate_rotation_angle(name, source_pose, target_pose):
    source_rot = source_pose[:3]
    target_rot = target_pose[:3]
    
    source_quat = R.from_rotvec(source_rot).as_quat()
    target_quat = R.from_rotvec(target_rot).as_quat()
    
    relative_quat = R.from_quat(target_quat) * R.from_quat(source_quat).inv()
    
    angle = R.from_quat(relative_quat.as_quat()).magnitude() * 180 / np.pi
    
    print(name, angle)
    return angle % 360

def create_annotation(datapath, exclude_video_id=''):
    annotations = []
    processed_videos = 0
    skipped_pairs = 0
    skipped_videos = 0
    
    train_path = os.path.join(datapath)
    
    if not os.path.exists(train_path):
        print(f"Error: The path {train_path} does not exist.")
        return annotations

    video_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print(f"Found {len(video_ids)} potential video directories.")

    for video_id in video_ids:
        if video_id == exclude_video_id:
            print(f"Skipping video ID: {exclude_video_id}")
            continue
        
        video_path = os.path.join(train_path, video_id)
        rotate_resize_path = os.path.join(video_path, 'extract_resize')
        intrinsics_path = os.path.join(video_path, 'extract_intrinsic_not_rotate')
        traj_file = os.path.join(video_path, 'lowres_wide.traj')
        
        if not all(os.path.exists(path) for path in [rotate_resize_path, intrinsics_path]):
            print(f"Skipping video ID {video_id}: Missing required directories")
            skipped_videos += 1
            continue

        if not os.path.exists(traj_file):
            print(f"Skipping video ID {video_id}: Missing traj file")
            skipped_videos += 1
            continue
        
        try:
            pose_data = process_traj_file(traj_file)
        except Exception as e:
            print(f"Error processing traj file for video ID {video_id}: {e}")
            skipped_videos += 1
            continue
        
        png_files = sorted([f for f in os.listdir(rotate_resize_path) if f.endswith('.png')])
        for i in range(0, len(png_files) - 1, 2):
            source_image = png_files[i]
            target_image = png_files[i + 1]
            print("source_image")
            
            source_rotated_path = os.path.join(rotate_resize_path, source_image)
            target_rotated_path = os.path.join(rotate_resize_path, target_image)
            
            if not os.path.exists(source_rotated_path) or not os.path.exists(target_rotated_path):
                print(f"Skipping pair in video {video_id}: Missing rotated images")
                skipped_pairs += 1
                continue
            
            source_timestamp = float(source_image.split('_')[-1].replace('.png', ''))
            target_timestamp = float(target_image.split('_')[-1].replace('.png', ''))
            
            source_closest = find_closest_timestamp(source_timestamp, pose_data)
            target_closest = find_closest_timestamp(target_timestamp, pose_data)
            
            if source_closest is None or target_closest is None:
                print(f"Skipping pair in video {video_id}: No close pose data for timestamps")
                skipped_pairs += 1
                continue
            
            source_intrinsics_file = os.path.join(intrinsics_path, source_image.replace('.png', '.pincam'))
            target_intrinsics_file = os.path.join(intrinsics_path, target_image.replace('.png', '.pincam'))
            
            if not os.path.exists(source_intrinsics_file) or not os.path.exists(target_intrinsics_file):
                print(f"Skipping pair in video {video_id}: Missing intrinsics files")
                skipped_pairs += 1
                continue

            try:
                source_intrinsics = process_pincam_file(source_intrinsics_file)
                target_intrinsics = process_pincam_file(target_intrinsics_file)
            except Exception as e:
                print(f"Error processing intrinsics for pair in video {video_id}: {e}")
                skipped_pairs += 1
                continue
            
            source_camera_pose = pose_data[source_closest]['rotation'] + pose_data[source_closest]['translation']
            target_camera_pose = pose_data[target_closest]['rotation'] + pose_data[target_closest]['translation']
            
            rotation_angle = calculate_rotation_angle(f"Training/{video_id}/extract_resize/{source_image}", 
                                                      source_camera_pose, target_camera_pose)
            
            annotation = {
                "source_image": f"Training/{video_id}/extract_resize/{source_image}",
                "target_image": f"Training/{video_id}/extract_resize/{target_image}",
                "source_camera_pose": source_camera_pose,
                "target_camera_pose": target_camera_pose,
                "source_image_timestamp": source_closest,
                "target_image_timestamp": target_closest,
                "source_camera_intrinsic": [
                    [source_intrinsics['focal_length_x'], 0, source_intrinsics['principal_point_x']],
                    [0, source_intrinsics['focal_length_y'], source_intrinsics['principal_point_y']],
                    [0, 0, 1]
                ],
                "target_camera_intrinsic": [
                    [target_intrinsics['focal_length_x'], 0, target_intrinsics['principal_point_x']],
                    [0, target_intrinsics['focal_length_y'], target_intrinsics['principal_point_y']],
                    [0, 0, 1]
                ],
                "rotation_angle": round(rotation_angle, 2)
            }
            
            annotations.append(annotation)
        
        processed_videos += 1
        if processed_videos % 10 == 0:
            print(f"Processed {processed_videos} videos. Current annotations: {len(annotations)}")
    
    print(f"Total videos processed: {processed_videos}")
    print(f"Total videos skipped: {skipped_videos}")
    print(f"Total pairs skipped: {skipped_pairs}")
    return annotations

if __name__ == "__main__":
    base_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/Training"
    annotations = create_annotation(base_path)
    
    current_directory = os.getcwd()
    json_file_path = os.path.join(current_directory, 'annotations_not_rotate2.json')
    
    with open(json_file_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Annotations saved to {json_file_path}. Total pairs: {len(annotations)}")

    # 필터링 부분 추가
    input_json_path = json_file_path
    output_json_path = os.path.join(current_directory, 'annotations_not_rotate_filter.json')

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # 20도 이하의 rotation_angle을 가진 항목만 필터링
    filtered_data = [item for item in data if item['rotation_angle'] <= 20]

    # 제거된 항목 수 계산
    removed_count = len(data) - len(filtered_data)

    # 새로운 JSON 파일로 저장
    with open(output_json_path, 'w') as file:
        json.dump(filtered_data, file, indent=2)

    print(f"원본 데이터 항목 수: {len(data)}")
    print(f"필터링 후 데이터 항목 수: {len(filtered_data)}")
    print(f"제거된 항목 수: {removed_count}")
    print(f"필터링된 데이터가 {output_json_path}에 저장되었습니다.")