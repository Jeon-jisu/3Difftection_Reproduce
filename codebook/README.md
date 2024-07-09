# 🧑‍💻ARKIT-SCENES 데이터셋 선택적 다운로드 스크립트

이 스크립트는 ARKIT-SCENES 데이터셋에서 3Diffdetection 논문에서 언급된 방식으로 데이터를 샘플링하고 다운로드하는 도구입니다.

## 🧐 사전 요구사항

1. ARKITSCENES GitHub 저장소를 클론한 후, 이 스크립트를 최상위 디렉토리에 위치해야 합니다.
2. Python 3.x가 설치되어 있어야 합니다.
3. Git이 설치되어 있어야 합니다.

## 🔥 사용 방법

1. ARKITSCENES GitHub 저장소를 클론합니다

```git clone https://github.com/apple/ARKitScenes.git```
```cd ARKitScenes```

2. 클론한 저장소의 최상위 디렉토리에 `download_script.py` 파일을 위치시킵니다.

3. 다음 명령어를 사용하여 스크립트를 실행합니다:
```python download_script.py --download_dir <다운로드_경로> ```

여기서 `<다운로드_경로>`는 데이터셋을 저장할 디렉토리 경로입니다.

## 🔍️ 스크립트 설명

이 스크립트는 3Diffdetection 논문에서 언급된 Geometric ControlNet 학습을 위한 데이터셋을 생성합니다.

- 목적: 4만장의 RGB 이미지(256x256)와 해당 이미지의 intrinsics 및 extrinsics 정보를 다운로드
- 샘플링 방법:
1. 각 비디오 ID에 대해 타임스탬프를 4등분합니다.
2. 각 구간에서 특정 시간 차이 미만의 이미지 쌍을 선택합니다.
3. 각 비디오 ID당 4개의 이미지 쌍(총 8장의 이미지)을 선택합니다.
- 이 방법을 통해 같은 비디오 ID 내에서도 시간에 따른 다양한 객체와 장면을 포착할 수 있습니다.

## 🙈 주의사항

- 스크립트 실행 전 `--download_dir` 인자에 올바른 다운로드 경로를 지정했는지 확인하세요.
- 다운로드에는 상당한 시간이 소요될 수 있으며, 충분한 저장 공간이 필요합니다.