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
from cldm.cldm import ControlLDM, EpipolarWarpOperator

input_path = sys.argv[1]
output_path = sys.argv[2]

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


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
