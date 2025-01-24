from controlnet.cldm.cldm import ControlLDM, SemanticControlNet, ControlNet
from controlnet.ldm.modules.encoders.modules import FrozenCLIPEmbedder
from controlnet.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

import torch
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import torch.nn.functional as F
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
import yaml
import os
import logging
from torchvision.utils import save_image
logger = logging.getLogger(__name__)


class ControlLDMBackbone(Backbone):
    def __init__(self, cfg, input_shape,pretrained=True):
        super().__init__()
        # Load the full config
        self.save_counter = 0
        self.save_interval = 2320
        self.save_dir = 'feature_maps4'
        os.makedirs(self.save_dir, exist_ok=True)
        self.clip_embedder = FrozenCLIPEmbedder()
        with open(cfg.MODEL.CONTROLLDM.CONFIG_PATH, 'r') as f:
            full_config = yaml.safe_load(f)
        # Extract the necessary configurations
        model_config = full_config['model']['params']
        first_stage_config = model_config['first_stage_config']
        control_config = model_config['control_stage_config']['params']
        control_config.update({
            k: v for k, v in dict(cfg.MODEL.CONTROLLDM.CONTROL_STAGE).items() if k not in {'target', 'NAME'}
        })
        control_stage_config = {
            "target": "controlnet.cldm.cldm.ControlNet",
            "params": control_config
        }
        semantic_control_config = model_config['semantic_control_stage_config']['params']
        semantic_control_config.update({
            k: v for k, v in dict(cfg.MODEL.CONTROLLDM.SEMANTIC_CONTROL_STAGE).items() if k not in {'target', 'NAME'}
        })
        unet_config = model_config['unet_config']
        # SEMANTIC_CONTROL_STAGE에 target 추가
        semantic_control_stage_config = {
            "target": "controlnet.cldm.cldm.SemanticControlNet",
            "params": semantic_control_config
        }
        model_config.update({
            'control_stage_config': control_stage_config,
            'semantic_control_stage_config': semantic_control_stage_config,
            "unet_config": unet_config,
            "first_stage_config": first_stage_config,
            'control_key': cfg.MODEL.CONTROLLDM.CONTROL_KEY,
            'only_mid_control': cfg.MODEL.CONTROLLDM.ONLY_MID_CONTROL,
            "cond_stage_config": dict(cfg.MODEL.CONTROLLDM.COND_STAGE_CONFIG),
        })
        self.controlldm = ControlLDM(cfg.MODEL.CONTROLLDM.USE_GEOMETRIC_CONTROL,cfg.MODEL.CONTROLLDM.USE_SEMANTIC_CONTROL,**model_config,)
        if pretrained and cfg.MODEL.CONTROLLDM.WEIGHTS_PRETRAIN:
            self.load_pretrained_weights(cfg.MODEL.CONTROLLDM.WEIGHTS_PRETRAIN)
            logger.info("로드 완료")
        else:
            logger.info(f"로드 미완료, pretrained: {pretrained}, cfg.MODEL.CONTROLLDM.WEIGHTS_PRETRAIN: {cfg.MODEL.CONTROLLDM.WEIGHTS_PRETRAIN}")
        
        # TODO: 이 부분 그냥 Output feature 맞춰서 수정해주었는데 확인이 필요
        # Define output feature names and channels 
        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {
            "p2": 320, "p3": 320, "p4": 640, "p5": 1280, "p6": 1280
        }
        self._out_feature_strides = {
            "p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64
        }
        
    def load_pretrained_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # ControlLDM 모델의 state_dict와 로드된 state_dict의 키가 일치하는지 확인
        controlldm_state_dict = self.controlldm.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in controlldm_state_dict}
        
        # 가중치 로드
        self.controlldm.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pre-trained weights from {weights_path}")
    def save_feature_map(self, feature_map, original_image,  n, level):
        # CPU로 이동 및 numpy 배열로 변환
        feature_map_np = feature_map[0].detach().cpu().numpy()  # 배치의 첫 번째 이미지만 사용
        
        # 채널 차원을 평탄화 (C, H, W) -> (H*W, C)
        flattened = feature_map_np.reshape(feature_map_np.shape[0], -1).T
        
        # K-means 클러스터링 수행 (k=5)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(flattened)
        
        # 클러스터 레이블을 원래 형태로 재구성
        cluster_labels = kmeans.labels_.reshape(feature_map_np.shape[1], feature_map_np.shape[2])
        
        # 클러스터 중심값으로 이미지 생성
        cluster_centers = kmeans.cluster_centers_
        clustered_image = cluster_centers[cluster_labels]
        
        # 정규화
        normalized = (clustered_image - clustered_image.min()) / (clustered_image.max() - clustered_image.min())
        
        # RGB 이미지로 변환 (첫 3개 채널 사용 또는 채널 평균)
        if normalized.shape[2] >= 3:
            rgb_image = (normalized[:, :, :3] * 255).astype(np.uint8)
        else:
            rgb_image = (np.mean(normalized, axis=2)[:, :, np.newaxis] * 255).astype(np.uint8)
            rgb_image = np.repeat(rgb_image, 3, axis=2)
        
        # 이미지 저장
        img = Image.fromarray(rgb_image)
        img.save(os.path.join(self.save_dir, f'feature_map_{level}_{self.save_counter}.png'))
        
        # 원본 이미지 저장
        save_image(original_image[0], os.path.join(self.save_dir, f'original_image_{self.save_counter}.png'), normalize=True)
        # original_image_np = original_image[0].permute(1, 2, 0).cpu().numpy()
        # original_image_np = (original_image_np * 255).astype(np.uint8)
        # original_img = Image.fromarray(original_image_np)
        # original_img.save(os.path.join(self.save_dir, f'original_image_{self.save_counter}.png'))

    def forward(self, x):
        batch_size = x.shape[0]
        desired_timestep = 261
        dummy_timesteps = torch.full((batch_size,), desired_timestep, device=x.device, dtype=torch.long)
        # dummy_timesteps = torch.zeros(batch_size, device=x.device)
        
        # 빈 문자열 인코딩
        empty_string = ""
        with torch.no_grad():
            dummy_context = self.clip_embedder.encode([empty_string] * batch_size)
        # VAE 인코딩 수정
            x_encoded = self.controlldm.first_stage_model.encode(x)
            if isinstance(x_encoded, DiagonalGaussianDistribution):
                x_encoded = x_encoded.sample()  # 분포에서 샘플 추출
            x_encoded = x_encoded.detach()
        # cond 딕셔너리 생성
        cond = {
            "c_crossattn": [dummy_context],
            "c_concat": [x],  # control_hint가 없으면 입력 x를 사용
            "source_pose": None,
            "target_pose": None,
            "source_intrinsic": None,
            "target_intrinsic": None
        }
        # Stable Diffusion의 feature maps 추출
        sd_features = []
        h = x_encoded

        eps, decoder_features, combined_encoder_features = self.controlldm.apply_model(x_encoded, dummy_timesteps, cond, return_intermediate=True)
        # 여기서 decoder_features와 geo_features를 결합하거나 처리하여 outputs 생성
        outputs = {}
        selected_layers = [0,3,6,9,11]
        for i, feature_name in enumerate(self._out_features):
            if i < len(selected_layers):
                outputs[feature_name] = combined_encoder_features[selected_layers[i]]
            else:
                # p6의 경우 p5와 동일
                outputs[feature_name] = F.max_pool2d(outputs[self._out_features[i-1]], kernel_size=2, stride=2)
        # 특정 간격으로 feature map 저장
        self.save_counter += 1
        if self.save_counter % self.save_interval == 0:
            self.save_feature_map(outputs['p2'],x, 4, 'p2')  # 4x4
            self.save_feature_map(outputs['p3'],x, 8, 'p3')  # 8x8
            self.save_feature_map(outputs['p4'],x, 16, 'p4')  # 16x16
            self.save_feature_map(outputs['p5'],x, 32, 'p5')   # 32x32
            self.save_feature_map(outputs['p6'],x, 32, 'p6')   # 32x32

        return outputs
    
@BACKBONE_REGISTRY.register()
def build_controlldm_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    imagenet_pretrain = cfg.MODEL.WEIGHTS + cfg.MODEL.WEIGHTS_PRETRAIN != '' #cfg에서 가중치 경로가 설정되어있지 않다면 imagenet pretrain을 사용하려나봄.
    bottom_up = ControlLDMBackbone(cfg,input_shape,pretrained = imagenet_pretrain)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone