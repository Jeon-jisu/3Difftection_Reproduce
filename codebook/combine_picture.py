import os
from PIL import Image, ImageDraw, ImageFont
import math

def combine_images(directory, start_epoch, end_epoch):
    # 결과물을 저장할 'combine' 폴더 경로 설정
    combine_dir = os.path.join(os.path.dirname(directory), 'combine')
    
    # 'combine' 폴더가 없으면 생성
    if not os.path.exists(combine_dir):
        os.makedirs(combine_dir)
    
    # 디렉토리 내의 모든 폴더를 가져옵니다.
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # 폴더명을 기준으로 그룹화합니다.
    groups = {}
    for folder in folders:
        if folder.startswith('epoch-'):
            parts = folder.split('_')
            if len(parts) >= 3:
                key = '_'.join(parts[1:])
                if key not in groups:
                    groups[key] = []
                groups[key].append(folder)
    
    # 5단위로 나누어 범위 생성
    ranges = []
    for i in range(start_epoch, end_epoch + 1, 5):
        ranges.append((i, min(i + 4, end_epoch)))
    
    # 각 그룹과 범위에 대해 이미지를 결합합니다.
    for key, group in groups.items():
        # 에폭 순서대로 정렬
        group.sort(key=lambda x: int(x.split('-')[1].split('_')[0]))
        
        for range_start, range_end in ranges:
            # 지정된 범위의 에폭만 선택
            selected_group = [folder for folder in group if range_start <= int(folder.split('-')[1].split('_')[0]) <= range_end]
            
            if len(selected_group) < (range_end - range_start + 1):
                print(f"Skipping {key} for range {range_start}-{range_end}: Not enough epochs")
                continue
            
            # 첫 번째 이미지를 열어 크기를 확인합니다.
            first_image_path = os.path.join(directory, selected_group[0], 'combined.png')
            with Image.open(first_image_path) as img:
                width, height = img.size
            
            # 결과 이미지를 생성합니다.
            result = Image.new('RGB', (width * len(selected_group), height))
            draw = ImageDraw.Draw(result)
            font = ImageFont.load_default()
            
            # 각 이미지를 결과에 붙입니다.
            for i, folder in enumerate(selected_group):
                image_path = os.path.join(directory, folder, 'combined.png')
                with Image.open(image_path) as img:
                    result.paste(img, (i * width, 0))
                    # 에폭 번호를 추가합니다.
                    epoch_num = int(folder.split('-')[1].split('_')[0])
                    draw.text((i * width + 5, 5), f"Epoch {epoch_num}", font=font, fill=(255, 0, 0))
            
            # 결과를 'combine' 폴더에 저장합니다.
            output_filename = f"combine_epoch-{range_start}to{range_end}_{key}.png"
            result.save(os.path.join(combine_dir, output_filename))
            print(f"Saved: {output_filename} in {combine_dir}")

# 사용 예:
combine_images('/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/image_log/train_66_only_2_warp_2_5_v2/train', 80, 154)