import os
from collections import defaultdict

def count_images_in_folders(root_path):
    image_extensions = ('.png')
    counts = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == 'extract_resize':
            video_id = os.path.basename(os.path.dirname(dirpath))
            image_count = sum(1 for f in filenames if f.lower().endswith(image_extensions))
            counts[video_id] = image_count

    return counts

# 경로를 지정해주세요
root_path = '{count하고싶은 사진 폴더 경로 ex. Traing/}'

image_counts = count_images_in_folders(root_path)

# 결과 출력
for video_id, count in image_counts.items():
    print(f"Video ID: {video_id}, Image count: {count}")

# 총 이미지 개수
total_images = sum(image_counts.values())
print(f"\nTotal images across all folders: {total_images}")