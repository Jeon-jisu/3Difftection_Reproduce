import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def preprocess_geometric(data):
    """
    Preprocess data for Geometric ControlNet.
    Assumes input is a numpy array of shape (H, W, C).
    """
    # Normalize the data
    data = (data - np.mean(data)) / np.std(data)

    # Convert to torch tensor and change dimension order
    data = torch.from_numpy(data).float().permute(2, 0, 1)

    # Add batch dimension
    data = data.unsqueeze(0)

    return data


def preprocess_semantic(image_path):
    """
    Preprocess data for Semantic ControlNet.
    Assumes input is a path to an image file.
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open image and apply transformations
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image


def preprocess_head_detection(data):
    """
    Preprocess data for 3D Head Detection.
    Assumes input is a numpy array of shape (H, W, C).
    """
    # Resize the data to a fixed size (e.g., 224x224)
    data = np.resize(data, (224, 224, 3))

    # Normalize the data
    data = (data - np.mean(data)) / np.std(data)

    # Convert to torch tensor and change dimension order
    data = torch.from_numpy(data).float().permute(2, 0, 1)

    # Add batch dimension
    data = data.unsqueeze(0)

    return data


def augment_data(image, target):
    """
    Apply data augmentation techniques.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = transforms.functional.hflip(image)
        target = transforms.functional.hflip(target)

    # Random rotation
    angle = np.random.uniform(-10, 10)
    image = transforms.functional.rotate(image, angle)
    target = transforms.functional.rotate(target, angle)

    # Random color jitter
    color_jitter = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    )
    image = color_jitter(image)

    return image, target


def normalize_3d_coords(coords):
    """
    Normalize 3D coordinates to a unit cube.
    """
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    normalized_coords = (coords - min_coords) / (max_coords - min_coords)
    return normalized_coords


def preprocess_batch(batch, model_type):
    """
    Preprocess a batch of data based on the model type.
    """
    if model_type == "geometric":
        return torch.stack([preprocess_geometric(item) for item in batch])
    elif model_type == "semantic":
        return torch.stack([preprocess_semantic(item) for item in batch])
    elif model_type == "head_detection":
        return torch.stack([preprocess_head_detection(item) for item in batch])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
