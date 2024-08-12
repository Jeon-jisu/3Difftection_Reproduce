import json
import os
import random

def reduce_image_pairs_to_target(json_file_path, target_count=40592):
    # JSON 파일 읽기
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    print(f"Total image pairs before reduction: {len(data)}")

    # 제거할 쌍의 수 계산
    pairs_to_remove = len(data) - target_count // 2
    pairs_to_remove = max(pairs_to_remove, 0)  # 음수가 되지 않도록

    # 무작위로 쌍 선택 및 제거
    random.shuffle(data)
    removed_pairs = data[:pairs_to_remove]
    remaining_pairs = data[pairs_to_remove:]

    print(f"Pairs removed: {len(removed_pairs)}")
    print(f"Pairs remaining: {len(remaining_pairs)}")

    # 제거된 이미지 파일 삭제 (선택적)
    for pair in removed_pairs:
        source_image = pair['source_image']
        target_image = pair['target_image']
        if os.path.exists(source_image):
            os.remove(source_image)
            print(f"Removed: {source_image}")
        if os.path.exists(target_image):
            os.remove(target_image)
            print(f"Removed: {target_image}")

    # 새로운 JSON 파일 저장
    new_json_path = json_file_path.replace('.json', '_reduced.json')
    with open(new_json_path, 'w') as file:
        json.dump(remaining_pairs, file, indent=2)

    print(f"Updated JSON saved to: {new_json_path}")
    print(f"Final image pair count: {len(remaining_pairs)}")

# JSON 파일 경로 지정
json_file_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/annotations_not_rotate_filter.json'

# 함수 실행
reduce_image_pairs_to_target(json_file_path)