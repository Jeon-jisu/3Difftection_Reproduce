import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class HeadDetection3D(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet50"):
        super(HeadDetection3D, self).__init__()

        # Load pre-trained ResNet backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(2048, 256)

        # 3D Head detection layers
        self.head_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.head_cls = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head_reg = nn.Conv2d(
            256, 12, kernel_size=1
        )  # 3D bounding box (x, y, z, w, h, d) + 3D orientation (3) + 3D center (3)

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)

        # Apply FPN
        fpn_features = self.fpn(features)

        # Detect heads in 3D
        head_features = self.head_conv(fpn_features[-1])
        cls_logits = self.head_cls(head_features)
        reg_preds = self.head_reg(head_features)

        return cls_logits, reg_preds


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for _ in range(3):  # Assuming 3 levels in the FPN
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            in_channels //= 2

    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[-1](x)
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(self.inner_blocks) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


def post_process(cls_logits, reg_preds, anchors):
    # Implement post-processing here (NMS, etc.)
    pass


# Example usage
if __name__ == "__main__":
    model = HeadDetection3D(num_classes=1)
    x = torch.randn(1, 3, 512, 512)
    cls_logits, reg_preds = model(x)
    print(f"Classification logits shape: {cls_logits.shape}")
    print(f"Regression predictions shape: {reg_preds.shape}")
