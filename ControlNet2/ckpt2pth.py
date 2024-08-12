import torch

# PyTorch Lightning 체크포인트 파일 경로
ckpt_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet/lightning_logs/version_1/checkpoints/epoch=2-step=37500.ckpt'
pth_path = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet/lightning_logs/version_1/checkpoints/epoch=2-step=37500.pth'

# 체크포인트 파일 로드
checkpoint = torch.load(ckpt_path)
# print("checkpoint['state_dict']",checkpoint['state_dict'].keys())
# # 모델의 상태 사전 추출
state_dict = checkpoint['state_dict']

# # 필요한 경우 상태 사전의 키를 변환
# # 일반적으로 PyTorch Lightning에서는 키에 'model.' 접두사가 붙으므로 이를 제거해야 할 수 있습니다.
new_state_dict = {}
for key, value in state_dict.items():
    new_state_dict[key] = value
    # print(new_key)

# 상태 사전을 .pth 파일로 저장
torch.save(new_state_dict, pth_path)
print(f"Model state_dict saved to {pth_path}")
