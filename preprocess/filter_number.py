import os
import glob

def remove_top_and_bottom_images(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == 'extract_rotate_resize':
            image_files = sorted(glob.glob(os.path.join(dirpath, '*.png')))  # png 파일 처리
            if len(image_files) <= 4:
                print(f"Warning: Folder {dirpath} has 4 or fewer images. Skipping.")
                continue
            
            to_remove = image_files[:2] + image_files[-2:]
            
            print(f"\nIn folder: {dirpath}")
            print("Files being removed:")
            for file in to_remove:
                print(os.path.basename(file))
                os.remove(file)
            print("Files have been removed.")

# 경로를 지정해주세요
root_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/Training'

# 함수 실행
remove_top_and_bottom_images(root_path)