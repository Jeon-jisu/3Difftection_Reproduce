import torch
import torch.nn.functional as F


def generate_camera_params(num_views=8):
    """Generate camera parameters for multiple views."""
    angles = torch.linspace(0, 2 * torch.pi, num_views)
    distances = torch.ones_like(angles) * 2.0  # Fixed distance from the object
    elevations = torch.ones_like(angles) * torch.pi / 4  # 45 degree elevation

    camera_params = torch.stack([angles, distances, elevations], dim=1)
    return camera_params


def apply_geometric_transform(image, geometric_features, camera_params):
    """Apply geometric transformation based on camera parameters."""
    batch_size, _, height, width = image.shape

    # Create a grid of coordinates
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, width), torch.linspace(-1, 1, height)
    )
    grid = (
        torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )

    # Apply transformation based on geometric features and camera parameters
    transformation = geometric_features.view(batch_size, 2, 3)
    grid = torch.bmm(
        grid.view(batch_size, -1, 2), transformation[:, :2, :2].transpose(1, 2)
    )
    grid = grid + transformation[:, :2, 2].unsqueeze(1)

    # Apply camera rotation and translation
    angle, distance, elevation = camera_params.unbind(1)
    rotation = torch.stack(
        [torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], dim=1
    ).view(batch_size, 2, 2)

    grid = torch.bmm(grid, rotation)
    grid = grid.view(batch_size, height, width, 2)

    # Apply the transformation to the image
    transformed_image = F.grid_sample(image, grid, align_corners=True)

    return transformed_image


def synthesize_views(image, geometric_features, semantic_features, num_views=8):
    """Synthesize multiple views of the input image."""
    camera_params = generate_camera_params(num_views)

    synthesized_views = []
    for i in range(num_views):
        # Apply geometric transformation
        transformed_image = apply_geometric_transform(
            image, geometric_features, camera_params[i : i + 1]
        )

        # Apply semantic features
        semantic_mask = torch.sigmoid(semantic_features)
        synthesized_view = (
            transformed_image * semantic_mask + (1 - semantic_mask) * image
        )

        synthesized_views.append(synthesized_view)

    return torch.cat(synthesized_views, dim=0)
