import os
import csv
import glob
import numpy as np
import time
import shutil


# CSV 파일 경로
csv_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/metadata.csv"

# CSV에서 sky_direction 읽기
def read_sky_directions():
    sky_directions = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sky_directions[row['video_id']] = row['sky_direction']
    return sky_directions

def adjust_and_rotate_params(width, height, fx, fy, cx, cy, direction):
    original_width, original_height = width, height
    new_width, new_height = 256, 256
    rotated_width, rotated_height = original_width, original_height
    rotated_fx, rotated_fy = fx, fy
    rotated_cx, rotated_cy = cx, cy
    
    # 회전 적용 (원본 크기 기준)
    # if direction == "Up":
    #     rotated_width, rotated_height = original_width, original_height
    #     rotated_fx, rotated_fy = fx, fy
    #     rotated_cx, rotated_cy = cx, cy
    # elif direction == "Right":
    #     rotated_width, rotated_height = original_height, original_width
    #     rotated_fx, rotated_fy = fy, fx
    #     rotated_cx, rotated_cy = original_height - cy, cx
    # elif direction == "Down":
    #     rotated_width, rotated_height = original_width, original_height
    #     rotated_fx, rotated_fy = fx, fy
    #     rotated_cx, rotated_cy = original_width - cx, original_height - cy
    # elif direction == "Left":
    #     rotated_width, rotated_height = original_height, original_width
    #     rotated_fx, rotated_fy = fy, fx
    #     rotated_cx, rotated_cy = cy, original_width - cx
    # else:
    #     raise ValueError(f"Unknown direction: {direction}")
    
    # 크기 조정
    new_fx = rotated_fx * (new_width / rotated_width)
    new_fy = rotated_fy * (new_height / rotated_height)
    new_cx = rotated_cx * (new_width / rotated_width)
    new_cy = rotated_cy * (new_height / rotated_height)
    
    return [new_width, new_height, new_fx, new_fy, new_cx, new_cy]

def process_pincam_files(input_dir, output_dir, sky_direction):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pincam_files = glob.glob(os.path.join(input_dir, '*.pincam'))
    
    for pincam_file in pincam_files:
        try:
            with open(pincam_file, 'r') as f:
                params = list(map(float, f.read().strip().split()))
            
            new_params = adjust_and_rotate_params(*params, sky_direction)
            
            new_filename = os.path.join(output_dir, os.path.basename(pincam_file))
            
            with open(new_filename, 'w') as f:
                f.write(' '.join(map(str, new_params)))
            
            print(f"Processed: {pincam_file} -> {new_filename}")
        except Exception as e:
            print(f"Error processing {pincam_file}: {str(e)}")

def process_all_video_folders(raw_dir):
    processed_folders = set()
    sky_directions = read_sky_directions()

    while True:
        video_folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
        
        for video_folder in video_folders:
            if video_folder in processed_folders:
                continue

            input_dir = os.path.join(raw_dir, video_folder, 'extract_intrinsics')
            new_output_dir = os.path.join(raw_dir, video_folder, 'extract_intrinsic_not_rotate')
            
            if os.path.exists(input_dir):
                sky_direction = sky_directions.get(video_folder)
                if sky_direction:
                    try:
                        print(f"Processing folder: {video_folder}")
                        process_pincam_files(input_dir, new_output_dir, sky_direction)
                        processed_folders.add(video_folder)
                    except Exception as e:
                        print(f"Error processing folder {video_folder}: {str(e)}")
                else:
                    print(f"Skipping folder {video_folder}: No sky direction found")
            else:
                print(f"Skipping folder {video_folder}: No extract_intrinsics directory found")
        
        print("Completed a cycle. Waiting before next check...")
        time.sleep(300)  # 5분 대기

base_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/Training"

process_all_video_folders(base_path)