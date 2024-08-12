import json

def read_first_5_items_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        if isinstance(data, list):
            for item in data[:5]:
                print(item)
        elif isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:
                    break
                print(f"{key}: {value}")

# 사용 예
file_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/up_annotations.json'
read_first_5_items_json(file_path)