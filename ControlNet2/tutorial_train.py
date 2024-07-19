from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint



# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()

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
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# for batch in dataloader:
#     print(batch.keys())
#     print("Image shape:", batch['image'].shape)
#     print("Text:", batch['text'] if 'text' in batch else "No text key")
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(max_epochs = 2, accelerator="gpu",devices=1, precision=32, callbacks=[logger,checkpoint_callback])

print(f"Using {trainer.device_ids} GPU(s):")
    
# device = trainer.strategy.root_device
# print(f"Using device: {device}")

# Train!
trainer.fit(model, dataloader)
