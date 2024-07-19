import sys
import os
import torch
from share import *
from cldm.model import create_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R
from cldm.cldm import ControlLDM, ModifiedControlNet, EpipolarWarpOperator

input_path = sys.argv[1]
output_path = sys.argv[2]

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


    
# class EpipolarWarpOperator(nn.Module):
#     def __init__(self):
#         super(EpipolarWarpOperator, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()

#     def axis_angle_to_rotation_matrix(self, axis_angle):
#         r = R.from_rotvec(axis_angle)
#         return torch.tensor(r.as_matrix(), dtype=torch.float32)
    
#     def skew_symmetric(v):
#         return torch.tensor([[0, -v[2], v[1]],
#                             [v[2], 0, -v[0]],
#                             [-v[1], v[0], 0]])
        
#     def forward(self, x, intrinsic_params, relative_pose):
#         # Compute the epipolar line lc
#         batch_size, _, height, width = x.size()
#         K_inv = torch.inverse(intrinsic_params)
        
#         # Extract rotation and translation from relative_pose
#         rotation_vector = relative_pose[:, :3]
#         translation_vector = relative_pose[:, 3:]

#         # Compute rotation matrices
#         Rn = torch.stack([self.axis_angle_to_rotation_matrix(rv) for rv in rotation_vector])
#         tn = translation_vector.unsqueeze(2) # [-0.295128,1.53793,0.235118] -> [[-0.295128],[1.53793],[0.235118]] (3) -> (3,1)

#         # 이미지의 모든 픽셀 좌표를 동차 좌표(homogeneous coordinates) 형식으로 생성.
#         # u는 x좌표. v는 y좌표 Matrix를 표현함. 
#         u, v = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
#         u, v = u.float().to(x.device), v.float().to(x.device)
#         # 모두 1로 구성된 3번째 차원 추가하여 동차좌표 (x,y,1) 구성
#         ones = torch.ones_like(u)
#         pixel_coords = torch.stack((u, v, ones), dim=2).view(-1, 3).transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1)
        
#         # E는 essential matrix, F는 fundamental matrix
#         E = torch.matmul(skew_symmetric(tn.squeeze(2)), Rn)
#         F = torch.matmul(torch.matmul(K_inv.transpose(1, 2), E), K_inv)
        
#         # E와 F로 Epipolar lines로 계산하기
#         lc = torch.matmul(K_inv.transpose(1, 2), torch.matmul(tn, Rn)).transpose(1, 2)
#         lc = torch.matmul(F, pixel_coords)

#         # Sample features along the epipolar line
#         sampled_features = []
#         for i in range(lc.size(1)):
#             pi = lc[:, i, :].view(batch_size, height, width)
#             sampled_features.append(F.grid_sample(x, pi.unsqueeze(2), mode='bilinear', align_corners=True))

#         # Aggregate features
#         aggregated_features = torch.mean(torch.stack(sampled_features, dim=0), dim=0)

#         # Apply convolution and activation
#         x = self.conv1(aggregated_features)
#         x = self.relu(x)
#         return x

# 기존 모델 로드
base_model = create_model(config_path='./models/cldm_v15v2.yaml')
only_mid_control = False
pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

# config에서 필요한 매개변수 추출
model_config_path = './models/cldm_v15v2.yaml'

with open(model_config_path, 'r') as file:
    config = yaml.safe_load(file)
    

# config에서 필요한 매개변수 추출
params = config['model']['params'] # num_timesteps_cond 1번 들어갔고.

# 수정된 ControlNet 초기화
model = ControlLDM(
    **params
)

# pretrained_weights에서 필요한 가중치를 가져와서 model에 로드
scratch_dict = model.state_dict()
target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
