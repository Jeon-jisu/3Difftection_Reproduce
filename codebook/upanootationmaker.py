import json
import csv
import os

# metadata.csv 파일에서 video_id와 sky_direction 정보 읽기
def read_metadata(file_path):
    up_videos = set()
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['sky_direction'] == 'Up':
                up_videos.add(row['video_id'])
    return up_videos

# annotation.json 파일 처리
def process_annotations(json_path, up_videos, output_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    filtered_annotations = []
    for item in annotations:
        source_image = item['source_image']
        video_id = source_image.split('/')[1]
        if video_id in up_videos:
            filtered_annotations.append(item)

    with open(output_path, 'w') as f:
        json.dump(filtered_annotations, f, indent=2)

# 메인 실행 부분
metadata_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/metadata.csv'
annotation_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/annotations.json'
output_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/up_annotations.json'

up_videos = read_metadata(metadata_path)
process_annotations(annotation_path, up_videos, output_path)

print(f"처리가 완료되었습니다. 결과가 {output_path}에 저장되었습니다.")