import torch
from models.geometric_controlnet import GeometricControlNet
from utils.data_loader import Omni3DDataset
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


def train_geometric_controlnet(config):
    device = torch.device(config["device"])

    # Load datasets
    train_dataset = Omni3DDataset(
        config["data"]["data_dir"], "ARKitScenes", "train", resolution=256
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Initialize models
    unet = UNet2DConditionModel.from_pretrained(config["stable_diffusion_path"])
    model = GeometricControlNet(
        unet,
        num_views=config["model"]["num_views"],
        aggregation_method=config["model"]["aggregation_method"],
        warp_last_n_stages=2,
        input_channels=config["model"]["input_channels"],
        output_channels=config["model"]["output_channels"],
        num_blocks=config["model"]["num_blocks"],
        base_channels=config["model"]["base_channels"],
        channel_multiplier=config["model"]["channel_multiplier"],
    )
    model.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["lr_scheduler"]["step_size"],
        gamma=config["training"]["lr_scheduler"]["gamma"],
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        for batch_idx, batch in enumerate(train_loader):
            # Prepare input
            target_view = batch["image"].to(device)
            camera_poses = batch["camera_pose"].to(device)
            camera_intrinsics = batch["camera_intrinsics"].to(device)  # 추가

            # Sample noise and add to images
            noise = torch.randn_like(target_view)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (target_view.shape[0],),
                device=target_view.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(target_view, noise, timesteps)

            # Forward pass
            noise_pred = model(noisy_images, timesteps, camera_poses, camera_intrinsics)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % config["logging"]["log_interval"] == 0:
                print(
                    f"Epoch [{epoch+1}/{config['training']['num_epochs']}], "
                    f"Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )
        # Step the learning rate scheduler
        scheduler.step()
        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"{config['save_path']}_epoch{epoch+1}.pth")

    # Save the model
    torch.save(model.state_dict(), config["save_path"])
