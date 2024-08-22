"""
실행시키는 방법
python tool_add_control_custom_epiagr.py ./models/v1-5-pruned.ckpt ./models/control_new.ckpt

# """
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
from cldm.cldm import ControlLDM, EpipolarWarpOperator, SemanticControlNet

input_path = sys.argv[1]
output_path = sys.argv[2]

# def get_node_name(name, parent_name):
#     if len(name) <= len(parent_name):
#         return False, ''
#     p = name[:len(parent_name)]
#     if p != parent_name:
#         return False, ''
#     return True, name[len(parent_name):]


# # 기존 학습된 모델 로드
# pretrained_model = create_model(config_path='./models/cldm_v15v3.yaml')
# pretrained_weights = torch.load(input_path)
# if 'state_dict' in pretrained_weights:
#     pretrained_weights = pretrained_weights['state_dict']


# # config에서 필요한 매개변수 추출
# model_config_path = './models/cldm_v15v3.yaml'

# with open(model_config_path, 'r') as file:
#     config = yaml.safe_load(file)
    

# # config에서 필요한 매개변수 추출
# params = config['model']['params'] # num_timesteps_cond 1번 들어갔고.

# # 수정된 ControlNet 초기화
# new_model = ControlLDM(
#     **params
# )

# # pretrained_weights에서 필요한 가중치를 가져와서 model에 로드
# new_dict = new_model.state_dict()
# sd_keys = [k for k in pretrained_weights.keys() if k.startswith('model.diffusion_model')]
# semantic_keys = [k for k in new_dict.keys() if k.startswith('semantic_control_model')]
# cratch_dict = new_model.state_dict()
# # print("Stable Diffusion keys:")
# # print(sd_keys[:10])  # 처음 10개의 키만 출력
# # print("\nSemantic ControlNet keys:")
# # print(semantic_keys[:10])  # 처음 10개의 키만 출력
# for name, param in new_model.named_parameters():
#     if name.startswith('control_model'):
#         # ControlNet 가중치 복사
#         if name in pretrained_weights:
#             new_dict[name].copy_(pretrained_weights[name])
#             param.requires_grad = False
#         else:
#             print(f"Initialized randomly (ControlNet): {name}")
#     elif name.startswith('model'):
#         # Stable Diffusion 가중치 복사
#         if name in pretrained_weights:
#             new_dict[name].copy_(pretrained_weights[name])
#             param.requires_grad = False
#         else:
#             print(f"Initialized randomly (Stable Diffusion): {name}")
#     elif name.startswith('semantic_control_model'):
#         # Semantic ControlNet 초기화
#         sd_name = name.replace('semantic_control_model', 'model.diffusion_model')
#         if sd_name in pretrained_weights:
#             new_dict[name].copy_(pretrained_weights[sd_name])
#             print(f"Copied from Stable Diffusion (Semantic ControlNet): {name}")
#         else:
#             print(f"Initialized randomly (Semantic ControlNet): {name}")
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(param)
#             elif 'bias' in name:
#                 nn.init.zeros_(param)
#         param.requires_grad = True
#     else:
#         print(f"Unexpected parameter: {name}")

# # 누락된 가중치 처리
# for name in pretrained_weights.keys():
#     if name.startswith('control_model') and name not in new_dict:
#         print(f"Pretrained weight not used: {name}")


# new_model.load_state_dict(new_dict)

# # 모델 저장
# torch.save(new_model.state_dict(), output_path)
# print('Done.')
import torch
import torch.nn as nn

# 기존 ControlNet 가중치 로드
controlnet_weights = torch.load('./checkpoints/train_66_only_2_warp_2_5_v2/model-epoch=79-train_loss=0.00.ckpt')
if 'state_dict' in controlnet_weights:
    controlnet_weights = controlnet_weights['state_dict']

# 원본 Stable Diffusion 가중치 로드
sd_weights = torch.load('/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/models/v1-5-pruned.ckpt')
if 'state_dict' in sd_weights:
    sd_weights = sd_weights['state_dict']
model_config_path = './models/cldm_v15v3.yaml'
with open(model_config_path, 'r') as file:
    config = yaml.safe_load(file)
params = config['model']['params']
# 새 모델 초기화
new_model = ControlLDM(**params)  # params는 모델 설정을 포함

new_dict = new_model.state_dict()
unexpected_keys = []
# 가중치 매핑 및 복사
for name, param in new_model.named_parameters():
    if name.startswith('control_model'):
        if name in controlnet_weights:
            new_dict[name].copy_(controlnet_weights[name])
            param.requires_grad = False
        else:
            print(f"Initialized from ControlNet: {name}")
    elif name.startswith('first_stage_model') or name.startswith('cond_stage_model'):
        if name in sd_weights:
            new_dict[name].copy_(sd_weights[name])
            param.requires_grad = False
        else:
            print(f"Initialized from Stable Diffusion: {name}")
    elif name.startswith('semantic_control_model'):
        if 'zero_convs' in name:
            # Zero convolution 층 처리
            nn.init.zeros_(param)
            print(f"Initialized to zero (Semantic ControlNet): {name}")
        elif 'input_hint_block' in name:
            # input_hint_block 처리
            if name.endswith('.weight'):
                nn.init.xavier_uniform_(param)
            elif name.endswith('.bias'):
                nn.init.zeros_(param)
            print(f"Initialized input_hint_block (Semantic ControlNet): {name}")
        else:
            sd_name = name.replace('semantic_control_model', 'model.diffusion_model')
            if sd_name in sd_weights:
                new_dict[name].copy_(sd_weights[sd_name])
            else:
                print(f"Initialized randomly (Semantic ControlNet): {name}")
        param.requires_grad = True
    elif name.startswith('model'):
        if name in sd_weights:
            new_dict[name].copy_(sd_weights[name])
            param.requires_grad = False
        else:
            print(f"Initialized from Stable Diffusion: {name}")
    else:
        unexpected_keys.append(name)

# 누락된 키 초기화
for name in new_dict.keys():
    if name not in new_dict:
        if 'zero_convs' in name:
            nn.init.zeros_(new_dict[name])
            print(f"Initialized to zero: {name}")
        elif 'input_hint_block' in name:
            if name.endswith('.weight'):
                nn.init.xavier_uniform_(new_dict[name])
            elif name.endswith('.bias'):
                nn.init.zeros_(new_dict[name])
            print(f"Initialized input_hint_block: {name}")
        elif 'weight' in name:
            nn.init.xavier_uniform_(new_dict[name])
            print(f"Initialized weight: {name}")
        elif 'bias' in name:
            nn.init.zeros_(new_dict[name])
            print(f"Initialized bias: {name}")

# 모델에 가중치 로드 (strict=False 사용)
new_model.load_state_dict(new_dict, strict=False)

print("Weights loaded successfully.")

# 가중치 상태 확인
for name, param in new_model.named_parameters():
    if name.startswith('semantic_control_model'):
        if 'zero_convs' in name:
            assert torch.all(param == 0), f"Zero Conv layer not initialized to zero: {name}"
        print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}, mean = {param.data.mean():.4f}, std = {param.data.std():.4f}")

# 모델 저장
torch.save(new_model.state_dict(), output_path)
print('Model saved successfully.')