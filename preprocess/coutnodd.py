import os
from collections import defaultdict

def count_odd_image_folders(root_path):
    image_extensions = ('.png',)
    odd_counts = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == 'extract_rotate_resize':
            video_id = os.path.basename(os.path.dirname(dirpath))
            image_count = sum(1 for f in filenames if f.lower().endswith(image_extensions))
            
            if image_count % 2 != 0:  # 이미지 개수가 홀수인 경우
                odd_counts[video_id] = image_count

    return odd_counts

# 경로를 지정해주세요
root_path = '{count하고싶은 사진 폴더 경로 ex. Traing/}'

odd_image_counts = count_odd_image_folders(root_path)

# 홀수 개의 이미지를 가진 폴더의 Video ID 개수
total_odd_video_ids = len(odd_image_counts)
print(f"Number of Video IDs with odd number of images: {total_odd_video_ids}")

# 결과 출력 (선택적)
print("\nVideo IDs with odd number of images:")
for video_id, count in odd_image_counts.items():
    print(f"Video ID: {video_id}, Image count: {count}")

# 홀수 이미지의 총 개수 (선택적)
total_odd_images = sum(odd_image_counts.values())
print(f"\nTotal odd images across all folders: {total_odd_images}")