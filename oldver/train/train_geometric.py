import torch
from models.geometric_controlnet import GeometricControlNet
from utils.data_loader import Omni3DDataset
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

def train_geometric_controlnet(model, dataloader, optimizer, config, device):
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        for batch in dataloader:
            # Omni3DDataset의 __getitem__이 반환하는 값들을 언패킹
            # 데이터를 device로 이동
            source_image = batch["source_image"].to(device)
            target_image = batch["target_image"].to(device)
            source_camera_pose = batch["source_camera_pose"].to(device)
            target_camera_pose = batch["target_camera_pose"].to(device)
            source_image_timestamp = batch["source_image_timestamp"].to(device)
            target_image_timestamp = batch["target_image_timestamp"].to(device)
            source_camera_intrinsic = batch["source_camera_intrinsic"].to(device)
            target_camera_intrinsic = batch["target_camera_intrinsic"].to(device)
            timestep = batch['timestep'].to(device)
            # 모델에 입력 전달
            # 이 모델은 GeometricControlNet 임. 
            output = model(
                source_image, 
                target_image, 
                timestep,
                source_camera_pose, 
                target_camera_pose, 
                source_image_timestamp,
                target_image_timestamp,
                source_camera_intrinsic, 
                target_camera_intrinsic,
            )
            
            # 손실 계산 및 역전파
            loss = calculate_loss(output, target_image)  # 손실 함수는 별도로 정의해야 합니다
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # device = torch.device(config["device"])

    # # Load loader
    # train_loader = dataloader

    # # Initialize models
    # unet = UNet2DConditionModel.from_pretrained(config["stable_diffusion_path"],subfolder="unet")

    # # 여기서 model은 main.py에서 넘겨준 Geometric controlnet instance를 말함. 
    # model = model(
    #     unet=unet,
    #     **config["model"]
    # )
    # model.to(device)

    # # Initialize optimizer and scheduler
    # optimizer = optimizer 
    # scheduler = StepLR(
    #     optimizer,
    #     step_size=config["training"]["lr_scheduler"]["step_size"],
    #     gamma=config["training"]["lr_scheduler"]["gamma"],
    # )
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, subfolder="scheduler")

    # # Training loop
    # for epoch in range(config["training"]["num_epochs"]):
    #     for batch_idx, batch in enumerate(train_loader):
    #         # Prepare input
    #         source_image = batch["source_image"].to(device)
    #         target_image = batch["target_image"].to(device)
    #         source_camera_pose = batch["source_camera_pose"].to(device)
    #         target_camera_pose = batch["target_camera_pose"].to(device)
    #         source_camera_intrinsic = batch["source_camera_intrinsic"].to(device)
    #         target_camera_intrinsic = batch["target_camera_intrinsic"].to(device)

    #         # Sample noise and add to images
    #         noise = torch.randn_like(target_view)
    #         timesteps = torch.randint(
    #             0,
    #             noise_scheduler.num_train_timesteps,
    #             (target_view.shape[0],),
    #             device=target_view.device,
    #         ).long()
    #         noisy_target_images = noise_scheduler.add_noise(target_image, noise, timesteps)

    #         # Forward pass
    #         noise_pred = model(
    #             source_image,
    #             noisy_target_images,
    #             timesteps,
    #             source_camera_pose,
    #             target_camera_pose,
    #             source_camera_intrinsic,
    #             target_camera_intrinsic
    #         )
    #         # Compute loss
    #         loss = F.mse_loss(noise_pred, noise)

    #         # Backward pass and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

            # Logging
            if (batch + 1) % config["logging"]["log_interval"] == 0:
                print(
                    f"Epoch [{epoch+1}/{config['training']['num_epochs']}], "
                    f"Step [{batch+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"{config['save_path']}_epoch{epoch+1}.pth")

    # Save the model
    torch.save(model.state_dict(), config["save_path"])
