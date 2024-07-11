import os
import json
import numpy as np
from collections import defaultdict

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
    # Filter out timestamps that are within the max_diff range
    close_timestamps = [ts for ts in pose_data.keys() if abs(ts - timestamp) <= max_diff]
    # print("close_timestamps",close_timestamps,"timestamp",timestamp)
    # If no timestamps are within the max_diff range, return None
    if not close_timestamps:
        return None
    
    # Find the closest timestamp among the filtered timestamps
    closest = min(close_timestamps, key=lambda x: abs(x - timestamp))
    
    return closest

def create_annotation(base_path, exclude_video_id='41048181'):
    annotations = []
    processed_videos = 0
    skipped_videos = 0
    
    train_path = os.path.join(base_path, 'processed', 'geometric', 'train')
    
    if not os.path.exists(train_path):
        print(f"Error: The path {train_path} does not exist.")
        return annotations

    video_ids = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print(f"Found {len(video_ids)} potential video directories.")

    for video_id in video_ids:
        if video_id == exclude_video_id:
            print(f"Skipping video ID: {exclude_video_id}")
            skipped_videos += 1
            continue
        
        video_path = os.path.join(train_path, video_id)
        extract_path = os.path.join(video_path, 'extract')
        intrinsics_path = os.path.join(video_path, 'extract_intrinsics')
        traj_file = os.path.join(video_path, 'lowres_wide.traj')
        
        if not all(os.path.exists(path) for path in [extract_path, intrinsics_path, traj_file]):
            print(f"Skipping video ID {video_id}: Missing required files or directories")
            skipped_videos += 1
            continue

        # Process traj file
        pose_data = process_traj_file(traj_file)
        
        # Group images by pairs
        png_files = sorted([f for f in os.listdir(extract_path) if f.endswith('.png')])
        image_pairs = []
        for i in range(0, len(png_files) - 1, 2):
            source_image = png_files[i]
            target_image = png_files[i + 1]
            source_timestamp = float(source_image.split('_')[-1].replace('.png', ''))
            target_timestamp = float(target_image.split('_')[-1].replace('.png', ''))
            print("source_timestamp",source_timestamp)
            source_closest = find_closest_timestamp(source_timestamp, pose_data)
            target_closest = find_closest_timestamp(target_timestamp, pose_data)
            
            if source_closest is not None and target_closest is not None:
                image_pairs.append((source_image, target_image, source_closest, target_closest))
            else:
                print(f"Skipping pair in video {video_id}: No close pose data for timestamps")
        
        # Process each pair
        for source_image, target_image, source_timestamp, target_timestamp in image_pairs:
            source_pose = pose_data[source_timestamp]
            target_pose = pose_data[target_timestamp]
            
            source_intrinsics_file = os.path.join(intrinsics_path, source_image.replace('.png', '.pincam'))
            target_intrinsics_file = os.path.join(intrinsics_path, target_image.replace('.png', '.pincam'))
            
            if not os.path.exists(source_intrinsics_file) or not os.path.exists(target_intrinsics_file):
                print(f"Skipping pair in video {video_id}: Missing intrinsics files")
                continue

            source_intrinsics = process_pincam_file(source_intrinsics_file)
            target_intrinsics = process_pincam_file(target_intrinsics_file)
            
            annotation = {
                "source_image": f"train/{video_id}/extract/{source_image}",
                "target_image": f"train/{video_id}/extract/{target_image}",
                "source_camera_pose": source_pose['rotation'] + source_pose['translation'],
                "target_camera_pose": target_pose['rotation'] + target_pose['translation'],
                "source_image_timestamp":source_timestamp,
                "target_image_timestamp":target_timestamp,
                "source_camera_intrinsic": [
                    [source_intrinsics['focal_length_x'], 0, source_intrinsics['principal_point_x']],
                    [0, source_intrinsics['focal_length_y'], source_intrinsics['principal_point_y']],
                    [0, 0, 1]
                ],
                "target_camera_intrinsic": [
                    [target_intrinsics['focal_length_x'], 0, target_intrinsics['principal_point_x']],
                    [0, target_intrinsics['focal_length_y'], target_intrinsics['principal_point_y']],
                    [0, 0, 1]
                ]
            }
            
            annotations.append(annotation)
        
        processed_videos += 1
        if processed_videos % 10 == 0:
            print(f"Processed {processed_videos} videos. Current annotations: {len(annotations)}")
    
    print(f"Total videos processed: {processed_videos}")
    print(f"Total videos skipped: {skipped_videos}")
    return annotations

# 메인 실행 부분
if __name__ == "__main__":
    base_path = "data"  # 실제 데이터 디렉토리 경로로 변경하세요
    annotations = create_annotation(base_path)
    
    # JSON 파일로 저장
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Annotations saved to annotations.json. Total pairs: {len(annotations)}")