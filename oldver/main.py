import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from models import GeometricControlNet, SemanticControlNet, HeadDetection3D
from utils import Omni3DDataset
from train.train_geometric import train_geometric_controlnet
# from train.train_semantic import train_semantic_controlnet
# from train.train_head_detection import train_head_detection
from inference.inference_3difftection import inference_3difftection
from diffusers import UNet2DConditionModel


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def get_dataset(dataset_type, config):
    if dataset_type == "geometric":
        return Omni3DDataset(
            data_dir=config["data"]["data_dir"],
            split="train",
            resolution=config["data"].get("resolution", 256)
        )
    # elif dataset_type == "semantic":
    #     return SemanticDataset(config["data_dir"], augment=config.get("augment", False))
    # elif dataset_type == "head_detection":
    #     return HeadDetectionDataset(
    #         config["data_dir"], augment=config.get("augment", False)
    #     )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb



def get_model(model_type, config):
    if model_type == "geometric":
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="unet"
        )
        print(f"UNet model size: {get_model_size(unet):.2f} MB")
        geometric_model = GeometricControlNet(**config["model"], unet=unet)
        print(f"GeometricControlNet model size: {get_model_size(geometric_model):.2f} MB")
        
        return geometric_model
    # elif model_type == "semantic":
    #     return SemanticControlNet(**config["model"])
    # elif model_type == "head_detection":
    #     return HeadDetection3D(**config["model"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    torch.cuda.empty_cache()
    config = load_config(args.config)
    device = torch.device(args.device)

    # Set up dataset and model
    dataset = get_dataset(args.model, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    print("모델 로드 전")
    print_gpu_memory()
    print("모델 로드 중")
    model = get_model(args.model, config)
    
    print_gpu_memory()
    print("GPU로 모델 이동")
    model = model.half()
    model = model.to(device)
    print_gpu_memory()
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"])

    # Train the model
    if args.model == "geometric":
        train_geometric_controlnet(model, dataloader, optimizer, config , device)
    # elif args.model == "semantic":
    #     train_semantic_controlnet(model, dataloader, optimizer, config, device)
    # elif args.model == "head_detection":
    #     train_head_detection(model, dataloader, optimizer, config, device)


def inference(args):
    config = load_config(args.config)
    device = torch.device(args.device)

    detections = inference_3difftection(args.image, args.config, device)
    print("Detected heads:", detections)


def main():
    parser = argparse.ArgumentParser(description="3Difftection")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Mode to run the script in",
    )
    parser.add_argument(
        "--model",
        choices=["geometric", "semantic", "head_detection"],
        help="Model to train (required for training)",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--image", type=str, help="Path to input image (required for inference)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.model is None:
            parser.error("--model is required when mode is 'train'")
        train(args)
    elif args.mode == "inference":
        if args.image is None:
            parser.error("--image is required when mode is 'inference'")
        inference(args)


if __name__ == "__main__":
    main()
