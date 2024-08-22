from controlnet.cldm.cldm import ControlledUnetModel, ControlNet, SemanticControlNet
from controlnet.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from controlnet.ldm.models.autoencoder import AutoencoderKL
from controlnet.ldm.modules.encoders.modules import FrozenCLIPEmbedder
import torch
import torch.nn.functional as F
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
import yaml

class ControlledUNetBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        # Initialize CLIP embedder
        self.clip_embedder = FrozenCLIPEmbedder()
        # Load model config
        with open(cfg.MODEL.CONTROLLEDUNET.CONFIG_PATH, 'r') as f:
            model_config = yaml.safe_load(f)
        self.vae = AutoencoderKL(model_config['model']['params']['first_stage_config']['params'])
        unet_config = model_config['model']['params']['unet_config']['params']
        self.unet = ControlledUnetModel(**unet_config)
        
        # Initialize ControlNet
        control_config = model_config['model']['params']['control_stage_config']['params']
        self.controlnet = ControlNet(**control_config)

        # Initialize SemanticControlNet
        semantic_control_config = model_config['model']['params']['semantic_control_stage_config']['params']
        self.semantic_controlnet = SemanticControlNet(**semantic_control_config)
        
        # ControlledUNet settings
        self.control_key = cfg.MODEL.CONTROLLEDUNET.CONTROL_KEY
        self.only_mid_control = cfg.MODEL.CONTROLLEDUNET.ONLY_MID_CONTROL
        self.control_scales = [1.0] * 13  # Assuming 13 layers of control

        # Define output feature names and channels
        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {
            "p2": 320, "p3": 640, "p4": 1280, "p5": 1280, "p6": 1280
        }
        self._out_feature_strides = {
            "p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64
        }
        # Load pre-trained weights if specified
        if cfg.MODEL.CONTROLLEDUNET.PRETRAINED_PATH:
            state_dict = torch.load(cfg.MODEL.CONTROLLEDUNET.PRETRAINED_PATH, map_location='cpu')
            self.unet.load_state_dict(state_dict, strict=False)

        # Freeze SD part if specified
        if cfg.MODEL.CONTROLLEDUNET.SD_LOCKED:
            for param in self.unet.parameters():
                param.requires_grad = False

    def forward(self, x, control_hint=None):
        batch_size = x.shape[0]
        print("ControlledUNetBackbone forward Input tensor (x) shape:", x.shape)
        dummy_timesteps = torch.zeros(batch_size, device=x.device)
        # Encode empty string using CLIP text encoder
        empty_string = ""
        hs = []
        # VAE 인코딩
        with torch.no_grad():
            print("VAE Encodeing 시작")
            encoded_x = self.vae.encode(x).sample()
            print("VAE encoded tensor (x) shape:", encoded_x.shape)
            dummy_context = self.clip_embedder.encode(empty_string)
            # dummy_context = torch.randn(batch_size, 77, 768, device=x.device)  # Assuming CLIP text embeddings
            # Process input through ControlNet
            print("geometric controlnet으로 들어가기 전")
            geometric_control =self.controlnet(x=encoded_x, hint=x, timesteps=dummy_timesteps, context=dummy_context, 
                                    source_pose=None, target_pose=None, 
                                    source_intrinsic=None, target_intrinsic=None)

            # h = x.type(self.unet.dtype)
            # for module in self.unet.input_blocks:
            #     h = module(h, emb, dummy_context)
            #     hs.append(h)
            # h = self.unet.middle_block(h, emb, dummy_context)
            # geometric_control = self.controlnet(x=x, hint=x, timesteps=dummy_timesteps, context=dummy_context)

            # geometric_control = self.controlnet(x=x, hint=x, timesteps=dummy_timesteps, context=dummy_context)
        
        # # Get output from Semantic ControlNet
        semantic_control = self.semantic_controlnet(x=x, hint=x, timesteps=dummy_timesteps, context=dummy_context)
        
        # # Combine the outputs of both ControlNets
        # combined_control = [g + s for g, s in zip(geometric_control, semantic_control)]
        # combined_control = [c * scale for c, scale in zip(combined_control, self.control_scales)]

        # # Forward pass through ControlledUnetModel
        # features = self.unet(x, timesteps=dummy_timesteps, context=dummy_context, control=combined_control, only_mid_control=self.only_mid_control)
        
        # # Adapt output to FPN format
        # outputs = {
        #     "p2": features[1],
        #     "p3": features[2],
        #     "p4": features[3],
        #     "p5": features[4],
        #     "p6": F.max_pool2d(features[4], kernel_size=2, stride=2)
        # }
        # 3, 6, 9, 12번째 feature map 선택
        selected_geometric = [geometric_control[i] for i in [2, 5, 8, 11]]
        selected_semantic = [semantic_control[i] for i in [2, 5, 8, 11]]
        
        combined_control = [g + s for g, s in zip(selected_geometric, selected_semantic)]
        combined_control = [c * scale for c, scale in zip(combined_control, self.control_scales[:4])]  # 4개의 scale만 사용

        # ControlledUnetModel을 통과
        features = self.unet(x, timesteps=dummy_timesteps, context=dummy_context, control=combined_control, only_mid_control=self.only_mid_control)
        
        # # 출력 구성
        # outputs = {
        #     "p3": combined_control[0],
        #     "p6": combined_control[1],
        #     "p9": combined_control[2],
        #     "p12": combined_control[3]
        # }
        outputs = {
            "p2": combined_control[0],
            "p3": combined_control[1],
            "p4": combined_control[2],
            "p5": combined_control[3],
            "p6": F.max_pool2d(combined_control[3], kernel_size=2, stride=2)
        }
        return outputs

@BACKBONE_REGISTRY.register()
def build_controlled_unet_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    bottom_up = ControlledUNetBackbone(cfg, input_shape)
    # in_features = cfg.MODEL.FPN.IN_FEATURES
    in_features = ["p2", "p3", "p4", "p5", 'p6']
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone