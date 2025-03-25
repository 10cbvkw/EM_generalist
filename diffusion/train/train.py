import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import argparse

from accelerate import Accelerator

import sys
sys.path.append('./')
from dataset import TifImageDataset

def get_gpu_memory():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)

        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)

        allocated_gb = allocated / (1024 ** 3)
        reserved_gb = reserved / (1024 ** 3)

        print(f"GPU: {gpu_name} (ID: {gpu_id})")
        print(f"Memory Allocated: {allocated_gb:.2f} GB")
        print(f"Memory Reserved:  {reserved_gb:.2f} GB")
    else:
        print("No CUDA GPUs are available")

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

parser = argparse.ArgumentParser(description='Process some parameters')
parser.add_argument('--data_path', type=str, help='output location')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--output_path', type=str, default='256_1e5', help='output location')
parser.add_argument('--load', type=bool, default=False, help='load from other pretrained checkpoint')
parser.add_argument('--load_epoch', type=int, default=0, help='checkpoint epoch')
parser.add_argument('--load_path', type=str, default='256_1e5', help='checkpoint path')
parser.add_argument('--image_size', type=int, default=128, help='image size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=1001, help='training epochs')

args = parser.parse_args()

class TrainingConfig:
    image_size = args.image_size
    train_batch_size = args.batch_size
    eval_batch_size = 8
    num_epochs = args.epoch
    gradient_accumulation_steps = 1
    learning_rate = args.lr
    save_image_epochs = 1
    save_model_epochs = 1
    save_and_sample_every = 1
    mixed_precision = "no"  
    output_dir = './exp/' + args.output_path
    overwrite_output_dir = True 
    seed = 0
    sample_batch_size = 8
    num_inference_steps = 1000
    ema = True
    ema_decay = 0.999

config = TrainingConfig()

writer = SummaryWriter()

dataset = TifImageDataset(args.data_path)
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Define the model
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    norm_eps=1e-6,
    downsample_padding=0,
    flip_sin_to_cos=False,
    dropout=0.1,
    freq_shift=1,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model parameters:', pytorch_total_params)

load = args.load
start_epoch = args.load_epoch

if load == True:
    model_dir = f'exp/{args.load_path}/weights.epoch_{start_epoch-1}.pt'
    model = torch.load(model_dir)
    if config.ema:
        model_ema = torch.load(f'exp/{args.load_path}/weights_ema.epoch_{start_epoch-1}.pt')

accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

if accelerator.is_main_process:
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
    accelerator.init_trackers("train_example")
    if config.ema and load == False:
        model_ema = deepcopy(model).to(torch.device('cuda:0'))
    if config.ema and load == True:
        model_ema = model_ema.to(torch.device('cuda:0'))   

# Prepare everything with Accelerator
model, optimizer, dataloader = accelerator.prepare(
    model, torch.optim.AdamW(model.parameters(), lr=config.learning_rate), dataloader
)

# Define the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Training loop
losses = []
for epoch in range(start_epoch, config.num_epochs):
    model.train()
    step = 0
    for batch in tqdm(dataloader, desc="Batches", leave=False):
        batch = batch.to(accelerator.device)

        noise = torch.randn_like(batch).to(accelerator.device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=accelerator.device).long()
        noisy_images = noise_scheduler.add_noise(batch.contiguous(), noise, timesteps)
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0] 
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        losses.append(loss.item())
        
        writer.add_scalar("Loss/train_steps", loss.item(), epoch * len(dataloader) + step)
        step += 1

        optimizer.step()
        optimizer.zero_grad()

        if config.ema and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            ema_update(unwrapped_model, model_ema, config.ema_decay)
    
    loss_last_epoch = sum(losses[-len(dataloader) :]) / len(dataloader)
    
    if accelerator.is_main_process:
        print(f"Epoch:{epoch}, loss: {loss_last_epoch:.5f}")

    if epoch % config.save_model_epochs == 0 and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model, config.output_dir + '/weights.epoch_' + str(epoch) + '.pt')
        if config.ema:
            torch.save(model_ema, config.output_dir + '/weights_ema.epoch_' + str(epoch) + '.pt')
        torch.save(optimizer, config.output_dir + '/optimizers.epoch_' + str(epoch) + '.pt')
    
accelerator.end_training()
