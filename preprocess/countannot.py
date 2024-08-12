import json

def count_json_elements(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if isinstance(data, list):
            count = len(data)
            print(f"JSON 파일의 요소 개수: {count}")
        elif isinstance(data, dict):
            count = len(data.keys())
            print(f"JSON 파일의 키 개수: {count}")
        else:
            print("JSON 파일이 리스트나 딕셔너리 형식이 아닙니다.")
    
    except json.JSONDecodeError:
        print("유효하지 않은 JSON 파일입니다.")
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")

# 사용 예시
json_file_path = '{count하고싶은 annotation 파일 경로}'  # JSON 파일 경로를 여기에 입력하세요
count_json_elements(json_file_path)