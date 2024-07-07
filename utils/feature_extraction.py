import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = base_model
        self.features = nn.ModuleList()

        # Assuming base_model is a CNN, we extract features from multiple layers
        for name, module in self.base_model.named_children():
            if isinstance(module, nn.Conv2d):
                self.features.append(module)
            elif isinstance(module, nn.Sequential):
                for sub_module in module:
                    if isinstance(sub_module, nn.Conv2d):
                        self.features.append(sub_module)

    def forward(self, x):
        feature_maps = []
        for feature in self.features:
            x = feature(x)
            feature_maps.append(x)
        return feature_maps


def extract_feature_maps(synthesized_views, model):
    """Extract feature maps from synthesized views using the given model."""
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()

    all_feature_maps = []
    with torch.no_grad():
        for view in synthesized_views:
            feature_maps = feature_extractor(view)
            all_feature_maps.append(feature_maps)

    return all_feature_maps


def aggregate_feature_maps(feature_maps_list):
    """Aggregate feature maps from multiple views."""
    aggregated_maps = []
    for level_maps in zip(*feature_maps_list):
        aggregated = torch.mean(torch.stack(level_maps), dim=0)
        aggregated_maps.append(aggregated)
    return aggregated_maps
