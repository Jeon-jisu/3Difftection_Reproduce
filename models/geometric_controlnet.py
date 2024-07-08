import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch.nn.functional as F


class EpipolarWarpOperator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 실제 구현에서는 더 복잡한 로직이 필요할 수 있습니다
        self.warp_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, features, source_pose, target_pose):
        # 여기서는 간단한 구현을 보여줍니다. 실제로는 epipolar geometry를 사용해야 합니다
        pose_diff = target_pose - source_pose
        warped_features = self.warp_conv(features + pose_diff.view(-1, 7, 1, 1))
        return warped_features


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
        num_control_channels=7,
        num_views=3,
        aggregation_method="attention",
        warp_last_n_stages=2,
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
            [self._make_control_block(num_control_channels + 4, 320)]  # 4 for RGBD
            + [self._make_control_block(320, 320) for _ in range(11)]
        )

        # Zero convolutions
        self.zero_convs_in = nn.ModuleList(
            [self._make_zero_conv(320) for _ in range(12)]
        )
        self.zero_convs_out = nn.ModuleList(
            [self._make_zero_conv(320) for _ in range(12)]
        )

        # Epipolar Warp Operator
        self.epipolar_warp = EpipolarWarpOperator(320)

        # Aggregator
        self.aggregator = Aggregator(320, method=aggregation_method)

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
        condition_view,
        target_view,
        timestep,
        camera_poses,
        encoder_hidden_states=None,
    ):
        batch_size, _, height, width = target_view.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=x.device)

        # Extract features from target view using frozen SD encoder
        with torch.no_grad():
            target_features = self.unet.down_blocks(
                target_view, timestep, encoder_hidden_states=encoder_hidden_states
            )

        control_outputs = []
        for view_idx in range(self.num_views):
            camera_pose = (
                camera_poses[:, view_idx]
                .view(batch_size, -1, 1, 1)
                .repeat(1, 1, height, width)
            )
            x = torch.cat([condition_view[:, view_idx], camera_pose], dim=1)
            view_control_outputs = []
            for i, (control_layer, zero_conv_in, zero_conv_out) in enumerate(
                zip(self.control_layers, self.zero_convs_in, self.zero_convs_out)
            ):
                x = zero_conv_in(x)
                x = control_layer(x)
                if (
                    i >= num_stages - self.warp_last_n_stages
                ):  # 마지막 self.warp_last_n_stages 개수의 단계에서만 warping 적용
                    x = self.epipolar_warp(
                        x,
                        camera_poses[:, view_idx],
                        camera_poses[:, (view_idx + 1) % self.num_views],
                    )
                x = zero_conv_out(x)
                view_control_outputs.append(x)
            control_outputs.append(view_control_outputs)

        # Epipolar warping and aggregation (마지막 두 단계에만 적용)
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
    output = model(x, timestep, camera_poses)
    print(f"Output shape: {output.shape}")
