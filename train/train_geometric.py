import torch
from models.geometric_controlnet import GeometricControlNet
from utils.data_loader import Omni3DDataset
from diffusers import DDPMScheduler, UNet2DConditionModel


def train_geometric_controlnet(config):
    device = torch.device(config["device"])

    # Load datasets
    train_dataset = Omni3DDataset(
        config["data_dir"], "ARKitScenes", "train", resolution=256
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Initialize models
    unet = UNet2DConditionModel.from_pretrained(config["pretrained_model_path"])
    model = GeometricControlNet(unet)
    model.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Training loop
    for epoch in range(config["num_epochs"]):
        for batch in train_loader:
            # Prepare input
            clean_images = batch["image"].to(device)
            camera_poses = batch["camera_pose"].to(device)

            # Sample noise and add to images
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Empty text input
            empty_text = [""] * clean_images.shape[0]

            # Forward pass
            noise_pred = model(noisy_images, timesteps, camera_poses, empty_text)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), config["save_path"])
