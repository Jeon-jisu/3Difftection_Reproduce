import torch
import yaml
import argparse
from PIL import Image
import numpy as np
from torchvision.transforms import Resize, ToTensor, Normalize
from models.geometric_controlnet import GeometricControlNet
from models.semantic_controlnet import SemanticControlNet
from models.head_detection_3d import HeadDetection3D
from diffusers import StableDiffusionPipeline, DDIMScheduler


def load_model(model_class, config_path, checkpoint_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = model_class(**config["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    transform = Resize(target_size, Image.BILINEAR)
    image = transform(image)
    image = ToTensor()(image)
    image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    return image.unsqueeze(0)


def inference_3difftection(image_path, config_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load models
    sd_pipeline = StableDiffusionPipeline.from_pretrained(config["sd_model_path"]).to(
        device
    )
    sd_pipeline.scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)

    geometric_model = load_model(
        GeometricControlNet,
        config["geometric_config"],
        config["geometric_checkpoint"],
        device,
    )
    semantic_model = load_model(
        SemanticControlNet,
        config["semantic_config"],
        config["semantic_checkpoint"],
        device,
    )
    head_model = load_model(
        HeadDetection3D,
        config["head_detection_config"],
        config["head_detection_checkpoint"],
        device,
    )

    # Load and preprocess image
    image = preprocess_image(image_path).to(device)

    # Generate empty text embeddings
    empty_text = [""] * image.shape[0]
    text_inputs = sd_pipeline.tokenizer(
        empty_text,
        padding="max_length",
        max_length=sd_pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeddings = sd_pipeline.text_encoder(text_inputs.input_ids.to(device))[0]

    # Generate camera pose (this should be provided or estimated in a real scenario)
    camera_pose = torch.randn(
        image.shape[0], 7, device=device
    )  # 3 for translation, 4 for quaternion

    # Geometric ControlNet
    geometric_features = geometric_model(
        image, torch.tensor([999]).to(device), camera_pose, text_embeddings
    )

    # Semantic ControlNet
    semantic_features = semantic_model(
        image, torch.tensor([999]).to(device), text_embeddings
    )

    # Synthesize views
    synthesized_views = []
    for i in range(config["num_views"]):
        latents = torch.randn((1, 4, 64, 64), device=device)
        latents = latents * sd_pipeline.scheduler.init_noise_sigma

        for t in sd_pipeline.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = sd_pipeline.unet(
                    latents, t, encoder_hidden_states=text_embeddings
                ).sample
                noise_pred = geometric_features + semantic_features + noise_pred
                latents = sd_pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        synthesized_view = sd_pipeline.vae.decode(
            latents / sd_pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        synthesized_views.append(synthesized_view)

    # 3D Head Detection
    detections = []
    for view in synthesized_views:
        view_detections = head_model(view)
        detections.append(view_detections)

    # Perform NMS ensemble (this function needs to be implemented)
    final_detections = nms_ensemble(
        detections,
        iou_threshold=config["nms_iou_threshold"],
        score_threshold=config["nms_score_threshold"],
    )

    return final_detections


def nms_ensemble(detections, iou_threshold, score_threshold):
    # Implement NMS ensemble here
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3Difftection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()

    detections = inference_3difftection(args.image, args.config, args.device)
    print(f"Detected 3D heads: {detections}")
