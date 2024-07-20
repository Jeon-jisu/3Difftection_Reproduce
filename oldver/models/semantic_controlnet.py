import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class SemanticControlNet(nn.Module):
    def __init__(self, base_model_path, num_control_channels=3):
        super().__init__()

        # Load the base Stable Diffusion U-Net
        self.base_model = UNet2DConditionModel.from_pretrained(base_model_path)

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Create control net layers
        self.control_net = nn.ModuleList(
            [
                self._make_control_block(num_control_channels, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
                self._make_control_block(320, 320),
            ]
        )

        # Zero convolutions
        self.zero_convs = nn.ModuleList(
            [
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
                self._make_zero_conv(320),
            ]
        )

    def _make_control_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def _make_zero_conv(self, channels):
        return nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, timestep, context, control_signal):
        control_outputs = []
        for control_block in self.control_net:
            control_signal = control_block(control_signal)
            control_outputs.append(control_signal)

        base_output = self.base_model(
            x, timestep, encoder_hidden_states=context, return_dict=False
        )[0]

        for i, (control_output, zero_conv) in enumerate(
            zip(control_outputs, self.zero_convs)
        ):
            base_output += zero_conv(control_output)

        return base_output


# Example usage
if __name__ == "__main__":
    model = SemanticControlNet("stabilityai/stable-diffusion-2-1-base")
    x = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([500])
    context = torch.randn(1, 77, 768)
    control_signal = torch.randn(1, 3, 64, 64)  # Semantic segmentation map
    output = model(x, timestep, context, control_signal)
    print(f"Output shape: {output.shape}")
