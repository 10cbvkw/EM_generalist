import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
import cv2
from tqdm import tqdm

EMA = True
epoch = 999
image_size = 128
lr = 'EMDinverse'

if EMA:
    model_dir = f"exp/{lr}/weights_ema.epoch_{epoch}.pt"
else:
    model_dir = f"exp/{lr}/weights.epoch_{epoch}.pt"

model = torch.load(model_dir)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) 

model.eval()
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler) 

from torchvision.utils import save_image

num_samples = 64
image_size = image_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_inference_steps = 1000
with torch.no_grad():
    noisy_image = torch.randn((num_samples, model.config.in_channels, image_size, image_size), device=device)
    for t in tqdm(reversed(range(num_inference_steps))):
        timesteps = torch.full((num_samples,), t, dtype=torch.long, device=device)
        noise_pred = model(noisy_image, timesteps).sample
        noisy_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample 
    if EMA:
        save_image((noisy_image + 1) / 2, f'sample_ema_ep{epoch}_{lr}.jpg', nrow=8, padding=0)
    else:
        save_image((noisy_image + 1) / 2, f'sample_ep{epoch}_{lr}.jpg', nrow=8, padding=0)
