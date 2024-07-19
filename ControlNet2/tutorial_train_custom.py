from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import GeometryDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM, EpipolarWarpOperator, ModifiedControlNet
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger

# Configs
resume_path = './models/control_sd15_customv2.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# WandB 설정
wandb.init(project="3difftection", name="init_train_v3_Normalize_Change_onetoone")
wandb_logger = WandbLogger(project="3difftection")
 
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15v2.yaml').cpu()

model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # 모니터링할 메트릭
    dirpath='./checkpoints',  # 체크포인트가 저장될 디렉토리
    filename='model-{epoch:02d}-{train_loss:.2f}',  # 체크포인트 파일 이름
    save_top_k=1,  # 가장 좋은 k개의 체크포인트만 저장
    mode='min',  # 모니터링할 메트릭이 낮을수록 좋은 경우 'min'
    save_last=True  # 마지막 체크포인트도 저장
)

# Misc
base_dir = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/'
annotation_file = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/annotations.json'
dataset = GeometryDataset(annotation_file, base_dir)
dataloader = DataLoader(dataset, num_workers=127, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(max_epochs = 1000, accelerator="gpu",devices=1, precision=32, callbacks=[logger,checkpoint_callback],logger=wandb_logger)

print(f"Using {trainer.device_ids} GPU(s):")
    
# device = trainer.strategy.root_device
# print(f"Using device: {device}")

# Train!
trainer.fit(model, dataloader)
wandb.finish()