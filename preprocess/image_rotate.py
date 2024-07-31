# import os
# import csv
# from PIL import Image

# # 경로 설정
# base_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/new/raw/Training"
# video_id = "40777060"  # 처리할 특정 video_id
# src_folder = os.path.join(base_path, video_id, "extract")
# dst_folder = os.path.join(base_path, video_id, "extract_rotate_resize")

# # CSV 파일 경로
# csv_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/metadata.csv"

# # CSV에서 sky_direction 읽기
# sky_direction = None
# with open(csv_path, 'r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['video_id'] == video_id:
#             sky_direction = row['sky_direction']
#             break

# if sky_direction is None:
#     print(f"Sky direction for video {video_id} not found. Skipping processing.")
#     exit()

# # 회전 각도 설정 (반시계 방향 회전)
# rotation_degrees = {
#     "Up": 0,
#     "Right": 90,
#     "Down": 180,
#     "Left": 270
# }

# # 대상 폴더가 없으면 생성
# if not os.path.exists(dst_folder):
#     os.makedirs(dst_folder)

# # 이미지 처리 함수
# def process_image(src_path, dst_path, rotation):
#     with Image.open(src_path) as img:
#         rotated_img = img.rotate(rotation, expand=True)
#         resized_img = rotated_img.resize((256, 256), Image.LANCZOS)
#         resized_img.save(dst_path)

# # 파일 처리
# rotation = rotation_degrees.get(sky_direction, 0)
# for filename in os.listdir(src_folder):
#     if filename.endswith('.png'):
#         src_path = os.path.join(src_folder, filename)
#         dst_path = os.path.join(dst_folder, filename)
        
#         process_image(src_path, dst_path, rotation)
#         print(f"Processed: {filename}")

# print(f"Processing complete. Sky direction: {sky_direction}, Rotation: {rotation} degrees, Resized to 256x256")
import os
import csv
from PIL import Image
from tqdm import tqdm
import time

# 경로 설정
base_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/new/raw/Training"

# CSV 파일 경로
csv_path = "/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/metadata.csv"

# CSV에서 sky_direction 읽기
sky_directions = {}
with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sky_directions[row['video_id']] = row['sky_direction']

# 회전 각도 설정 (반시계 방향 회전)
rotation_degrees = {
    "Up": 0,
    "Right": 90,
    "Down": 180,
    "Left": 270
}

# 이미지 처리 함수
def process_image(src_path, dst_path, rotation):
    try:
        with Image.open(src_path) as img:
            rotated_img = img.rotate(rotation, expand=True)
            resized_img = rotated_img.resize((256, 256), Image.LANCZOS)
            resized_img.save(dst_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")
        return False

# 폴더 처리 함수
def process_folder(video_id):
    video_path = os.path.join(base_path, video_id)
    src_folder = os.path.join(video_path, "extract")
    dst_folder = os.path.join(video_path, "extract_rotate_resize")

    # extract 폴더가 없으면 건너뛰기
    if not os.path.exists(src_folder):
        print(f"Skipping {video_id}: No extract folder")
        return

    # 이미 처리된 폴더는 건너뛰기
    if os.path.exists(dst_folder):
        print(f"Skipping {video_id}: Already processed")
        return

    # sky_direction 확인
    sky_direction = sky_directions.get(video_id)
    if sky_direction is None:
        print(f"Skipping {video_id}: Sky direction not found")
        return

    # 대상 폴더 생성
    os.makedirs(dst_folder, exist_ok=True)

    # 파일 처리
    rotation = rotation_degrees.get(sky_direction, 0)
    processed_count = 0
    total_files = len([f for f in os.listdir(src_folder) if f.endswith('.png')])

    for filename in os.listdir(src_folder):
        if filename.endswith('.png'):
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            
            if process_image(src_path, dst_path, rotation):
                processed_count += 1

    if processed_count == total_files:
        print(f"Processed {video_id}. Sky direction: {sky_direction}, Rotation: {rotation} degrees, Resized to 256x256")
    else:
        print(f"Partially processed {video_id}. {processed_count}/{total_files} images processed.")

# 메인 처리 루프
while True:
    for video_id in tqdm(os.listdir(base_path)):
        if os.path.isdir(os.path.join(base_path, video_id)):
            process_folder(video_id)
    
    print("Completed a full cycle. Waiting before next cycle...")
    time.sleep(300)  # 5분 대기

print("All processing complete.")