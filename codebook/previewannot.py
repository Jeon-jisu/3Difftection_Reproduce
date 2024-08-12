import json

def preview_json_to_file(input_file_path, output_file_path, num_lines=5):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        preview = data[:num_lines]
    elif isinstance(data, dict):
        preview = dict(list(data.items())[:num_lines])
    
    with open(output_file_path, 'w') as out_file:
        json.dump(preview, out_file, indent=2)

    print(f"Preview has been saved to {output_file_path}")

# 사용 예
preview_json_to_file('/node_data/urp24s_jsjeon/3Difftection_Reproduce/testControlNet2/annotations_origin.json', '/node_data/urp24s_jsjeon/3Difftection_Reproduce/testControlNet2/annotations_origin2.json',num_lines=20)