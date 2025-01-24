import json
import matplotlib.pyplot as plt

# JSON 파일 읽기
with open('/node_data_2/urp24s_jsjeon/3Difftection_Reproduce/testControlNet2/annotations_origin.json', 'r') as file:
    data = json.load(file)

# 각도 추출 및 20도, 30도 초과 이미지 쌍 찾기
angles = []
over_20_degrees = []
over_30_degrees = []

for item in data:
    angle = item['rotation_angle']
    angles.append(angle)
    if angle > 30:
        over_30_degrees.append((angle, item["source_image"], item["target_image"]))
    if angle > 20:
        over_20_degrees.append((angle, item["source_image"], item["target_image"]))

# 히스토그램 생성
plt.figure(figsize=(10, 6))
plt.hist(angles, bins=36, range=(0, 180), edgecolor='black')
plt.title('Distribution of Rotation Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.xticks(range(0, 181, 30))
plt.grid(True, alpha=0.3)

# 그래프 저장
plt.savefig('rotation_angles_histogram_abc.png')
plt.close()

print("Histogram has been saved as 'rotation_angles_histogram22.png'")

# 기본 통계 출력
print(f"Total number of image pairs: {len(angles)}")
print(f"Mean angle: {sum(angles)/len(angles):.2f}")
print(f"Min angle: {min(angles):.2f}")
print(f"Max angle: {max(angles):.2f}")

# 20도 초과 이미지 쌍 출력
print(f"\nNumber of image pairs with rotation angle over 20 degrees: {len(over_20_degrees)}")

# 30도 초과 이미지 쌍 출력
print(f"\nNumber of image pairs with rotation angle over 30 degrees: {len(over_30_degrees)}")

print("\nImage pairs with rotation angle over 30 degrees:")
# for angle, source, target in sorted(over_30_degrees, key=lambda x: x[0], reverse=True):
#     print(f"Angle: {angle:.2f}, Source: {source}, Target: {target}")