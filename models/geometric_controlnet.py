import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch.nn.functional as F


class EpipolarWarpOperator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def axis_angle_to_rotation_matrix(self, axis_angle):
        theta = torch.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / (theta + 1e-8)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return (
            cos_theta * torch.eye(3, device=axis_angle.device).unsqueeze(0)
            + (1 - cos_theta) * axis.unsqueeze(-1) * axis.unsqueeze(-2)
            + sin_theta
            * torch.cross(
                torch.eye(3, device=axis_angle.device).unsqueeze(0),
                axis.unsqueeze(-1),
                dim=-1,
            )
        )

    def compute_fundamental_matrix(
        self, source_pose, target_pose, source_intrinsics, target_intrinsics
    ):
        # Convert axis-angle to rotation matrix
        R1 = self.axis_angle_to_rotation_matrix(source_pose[:, :3])
        R2 = self.axis_angle_to_rotation_matrix(target_pose[:, :3])
        t1, t2 = source_pose[:, 3:], target_pose[:, 3:]

        # Compute relative rotation and translation
        R = torch.matmul(R2, R1.transpose(1, 2))
        t = t2 - torch.matmul(R, t1.unsqueeze(-1)).squeeze(-1)

        # Compute essential matrix
        t_cross = torch.zeros_like(R)
        t_cross[:, 0, 1], t_cross[:, 0, 2] = -t[:, 2], t[:, 1]
        t_cross[:, 1, 0], t_cross[:, 1, 2] = t[:, 2], -t[:, 0]
        t_cross[:, 2, 0], t_cross[:, 2, 1] = -t[:, 1], t[:, 0]
        E = torch.matmul(t_cross, R)

        # Compute fundamental matrix
        F = torch.matmul(
            torch.matmul(torch.inverse(target_intrinsics).transpose(1, 2), E),
            torch.inverse(source_intrinsics),
        )
        return F

    def compute_epipolar_lines(self, points, F):
        # Compute epipolar lines for given points
        homogeneous_points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        epipolar_lines = torch.matmul(F, homogeneous_points.transpose(1, 2)).transpose(
            1, 2
        )
        return epipolar_lines

    def warp_features(
        self, features, source_pose, target_pose, source_intrinsics, target_intrinsics
    ):
        b, c, h, w = features.shape

        # Compute fundamental matrix
        F = self.compute_fundamental_matrix(
            source_pose, target_pose, source_intrinsics, target_intrinsics
        )

        # Create grid of pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(h, device=features.device),
            torch.arange(w, device=features.device),
        )
        points = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        points = points.unsqueeze(0).repeat(b, 1, 1)

        # Compute epipolar lines
        epipolar_lines = self.compute_epipolar_lines(points, F)

        # Compute closest points on epipolar lines
        a, b, c = (
            epipolar_lines[:, :, 0],
            epipolar_lines[:, :, 1],
            epipolar_lines[:, :, 2],
        )
        x = -(a * c) / (a**2 + b**2 + 1e-8)
        y = -(b * c) / (a**2 + b**2 + 1e-8)

        # Create sampling grid
        grid_x = (2.0 * x / (w - 1)) - 1.0
        grid_y = (2.0 * y / (h - 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b, h, w, 2)

        # Sample features using grid
        warped_features = F.grid_sample(
            features, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return warped_features

    def forward(
        self,
        features,
        source_pose,
        target_pose,
        source_intrinsics,
        target_intrinsics,
    ):
        warped_features = self.warp_features(
            features, source_pose, target_pose, source_intrinsics, target_intrinsics
        )
        refined_features = self.conv(warped_features)
        return refined_features


class Aggregator(nn.Module):
    def __init__(self, channels, method="attention"):
        super().__init__()
        self.method = method
        if method == "attention":
            self.attention = nn.MultiheadAttention(channels, 8)
            self.norm = nn.LayerNorm(channels)
        elif method not in ["mean", "max"]:
            raise ValueError(
                "Unsupported aggregation method. Choose 'attention', 'mean', or 'max'."
            )

    def forward(self, features_list):
        if self.method == "attention":
            # 특징들을 (seq_len, batch, channels) 형태로 변환
            features = torch.stack(features_list, dim=0)
            aggregated, _ = self.attention(features, features, features)
            return self.norm(features + aggregated).mean(dim=0)
        elif self.method == "mean":
            return torch.stack(features_list).mean(dim=0)
        elif self.method == "max":
            return torch.stack(features_list).max(dim=0)[0]


class GeometricControlNet(nn.Module):
    def __init__(
        self,
        unet,
        num_control_channels=6,  # num_control_channels는 카메라 포즈와 같은 제어 입력의 채널 수 (여기서는 rotation(3) + translation(3))
        num_views=3,
        aggregation_method="attention",
        warp_last_n_stages=2,
        input_channels=3,
        output_channels=64,
        num_blocks=5,
        base_channels=32,
        channel_multiplier=2,
    ):
        super().__init__()
        self.unet = unet
        self.num_views = num_views
        self.warp_last_n_stages = warp_last_n_stages

        # Freeze the base model parameters
        for param in self.unet.parameters():
            param.requires_grad = False

        # Create control net layers
        self.control_layers = nn.ModuleList(
            [
                self._make_control_block(
                    num_control_channels + input_channels, base_channels
                )
            ]
            + [
                self._make_control_block(
                    base_channels * (channel_multiplier**i),
                    base_channels * (channel_multiplier ** (i + 1)),
                )
                for i in range(num_blocks - 1)
            ]
        )

        # Zero convolutions
        self.zero_convs_in = nn.ModuleList(
            [
                self._make_zero_conv(base_channels * (channel_multiplier**i))
                for i in range(num_blocks)
            ]
        )
        self.zero_convs_out = nn.ModuleList(
            [
                self._make_zero_conv(base_channels * (channel_multiplier**i))
                for i in range(num_blocks)
            ]
        )

        # Epipolar Warp Operator
        self.epipolar_warp = EpipolarWarpOperator(
            base_channels * (channel_multiplier ** (num_blocks - 1))
        )

        # Aggregator
        self.aggregator = Aggregator(
            base_channels * (channel_multiplier ** (num_blocks - 1)),
            method=aggregation_method,
        )

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        return x * (self.noise_schedule[t] ** 0.5) + noise * (
            (1 - self.noise_schedule[t]) ** 0.5
        )

    def extract_features(self, x, timestep, text_embeds):
        # Stable Diffusion의 UNet을 사용하여 특징 추출
        down_block_res_samples, mid_block_res_sample = self.unet.down_blocks(
            x, timestep, encoder_hidden_states=text_embeds
        )
        return down_block_res_samples

    def _make_control_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def _make_zero_conv(self, channels):
        return nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(
        self,
        source_image, target_image, timestep, source_camera_pose, target_camera_pose, source_camera_intrinsic, target_camera_intrinsic
    ):
        condition = source_image
        camera_params = torch.cat([source_camera_pose, target_camera_pose, source_camera_intrinsic.flatten(), target_camera_intrinsic.flatten()], dim=1)
    
        batch_size, _, height, width = target_image.shape
        
        # 빈 텍스트에 대한 임베딩 생성 (모든 요소가 0인 텐서)
        empty_text_embeds = torch.zeros(batch_size, 77, 768, device=target_image.device)


        # Extract features from target view
        with torch.no_grad():
            target_features = self.unet.down_blocks(target_image, timestep, encoder_hidden_states=empty_text_embeds)


        control_outputs = []
        
        # 첫번째 이미지를 source camera로 잡고 나머지 이미지는 target camera로 처리하여 zero conv in, control layer, zero conv out을 통과
        
        for view_idx in range(self.num_views):
            if view_idx == 0:
                camera_pose = source_camera_pose
            else:
                camera_pose = target_camera_pose
            
            camera_pose = camera_pose.view(batch_size, -1, 1, 1).repeat(1, 1, height, width)
            x = torch.cat([condition, camera_pose], dim=1)
            view_control_outputs = []
            for i, (control_layer, zero_conv_in, zero_conv_out) in enumerate(
                zip(self.control_layers, self.zero_convs_in, self.zero_convs_out)
            ):
                x = zero_conv_in(x)
                x = control_layer(x)
                if i >= len(self.control_layers) - self.warp_last_n_stages:
                    x = self.epipolar_warp(
                        x,
                        source_camera_pose,
                        target_camera_pose,
                        source_camera_intrinsic,
                        target_camera_intrinsic,
                    )
                x = zero_conv_out(x)
                view_control_outputs.append(x)
            control_outputs.append(view_control_outputs)

        # Epipolar warping and aggregation (마지막 두 단계에만 적용) 순서가 이게 맞는지 확인 필요. 지금은 conv out -> feature warping
        
        num_stages = len(control_outputs[0])
        aggregated_controls = []
        for level in range(len(control_outputs[0])):
            if (
                level >= num_stages - self.warp_last_n_stages
            ):  # 마지막 두 단계에만 Epipolar warping 적용
                warped_features = []
                for view_idx in range(self.num_views):
                    source_pose = camera_poses[:, view_idx]
                    for target_idx in range(self.num_views):
                        if target_idx != view_idx:
                            target_pose = camera_poses[:, target_idx]
                            warped = self.epipolar_warp(
                                control_outputs[view_idx][level],
                                source_pose,
                                target_pose,
                            )
                            warped_features.append(warped)
                aggregated = self.aggregator(warped_features)
            else:
                # 이전 단계는 warping 없이 직접 사용
                aggregated = sum(co[level] for co in control_outputs) / self.num_views

            aggregated_controls.append(aggregated)

        # Combined Features는 aggregated_control과 target_features를 결합한 것

        combined_features = []
        for control, target in zip(aggregated_controls, target_features):
            combined_features.append(control + target)

        # 이렇게 결합된 특징들을 UNet의 업샘플링 경로(up_blocks)를 통과시키는 과정. 이 과정을 통해 점진적으로 해상도가 증가하며, 최종적으로 원본 입력 크기의 특징맵이 생성

        hidden_states = combined_features[-1]  # 가장 낮은 해상도의 특징맵으로 초기화

        # 각 up_block은 현재의 hidden_states와 combined_features에서 pop()한 skip connection을 입력으로 받아 처리
        for up_block in self.unet.up_blocks:
            hidden_states = up_block(hidden_states, combined_features.pop())

        # 최종 출력 생성
        return self.unet.conv_norm_out(hidden_states)


# Example usage
if __name__ == "__main__":
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="unet"
    )
    model = GeometricControlNet(unet, num_views=3, aggregation_method="mean")
    x = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([500])
    camera_poses = torch.randn(1, 3, 7)  # (batch, num_views, 7)
    camera_intrinsics = torch.randn(1, 3, 3, 3)
    output = model(x, timestep, camera_poses)
    print(f"Output shape: {output.shape}")
