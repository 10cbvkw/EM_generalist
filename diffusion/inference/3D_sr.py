import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='DPS 3D super-resolution')
parser.add_argument('--path', type=int, help='input path')
parser.add_argument('--gamma', type=float, help='step size')
parser.add_argument('--factor', type=int, help='fold factor for super-resolution')
parser.add_argument('--output_path', type=str, help='output path')
args = parser.parse_args()

epoch = 999
checkpoint = ''

path = args.path
output_path = args.output_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = f"../train/exp/{checkpoint}/weights_ema.epoch_{epoch}.pt"
model = torch.load(model_dir).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# Set model to evaluation mode
model.eval()
torch.set_grad_enabled(True)

def posterior_sample(model, scheduler, noised_volume, num_inference_steps=1000, volume_size=128, gamma=1, K=4, interpolate=4, slice1_=[0], slice2_=[1], slice3_=[2]):

    noised_volume = noised_volume.to(device)
    noisy_volume = torch.randn((1, model.config.in_channels, volume_size, volume_size, volume_size), device=device, requires_grad=True)

    for t in reversed(range(num_inference_steps)):

        timesteps = torch.full((1,), t, dtype=torch.long, device=device)
        noisy_volume = noisy_volume.clone()

        if (t % K) in slice2_:
            new_noisy_volume = noisy_volume.clone()
            for slice_2 in range(volume_size): 
                noisy_slice = noisy_volume[:, :, :, slice_2, :].clone()
                noised_slice = noised_volume[:, :, slice_2, :].clone()
                with torch.enable_grad():
                    noise_pred = model(noisy_slice, timesteps).sample
                    alpha_t = scheduler.alphas_cumprod[t]
                    noisy_slice_0 = (noisy_slice - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

                loss = torch.zeros(1, requires_grad=True, device=device)
            
                indices = torch.arange(0, volume_size, interpolate)
                noisy_slice_0_LR = noisy_slice_0[:, :, indices, :]
                
                difference = noised_slice - noisy_slice_0_LR 
                loss = loss + torch.linalg.norm(difference) 
                
                norm_grad = torch.autograd.grad(outputs=loss, inputs=noisy_slice)[0]
                gamma_ = gamma / loss.item() 
                noisy_slice = noisy_slice - norm_grad * gamma_ 
                noisy_slice = scheduler.step(noise_pred, t, noisy_slice).prev_sample 
                new_noisy_volume[:, :, :, slice_2, :] = noisy_slice
            
            noisy_volume = new_noisy_volume.clone()
        
        elif (t % K) in slice3_:
            new_noisy_volume = noisy_volume.clone()
            for slice_3 in range(volume_size): 
                noisy_slice = noisy_volume[:, :, :, :, slice_3].clone()
                noised_slice = noised_volume[:, :, :, slice_3].clone()
                with torch.enable_grad():
                    noise_pred = model(noisy_slice, timesteps).sample
                    alpha_t = scheduler.alphas_cumprod[t]
                    noisy_slice_0 = (noisy_slice - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

                loss = torch.zeros(1, requires_grad=True, device=device)
            
                indices = torch.arange(0, volume_size, interpolate)
                noisy_slice_0_LR = noisy_slice_0[:, :, indices, :]
                
                difference = noised_slice - noisy_slice_0_LR 
                loss = loss + torch.linalg.norm(difference) 
                
                norm_grad = torch.autograd.grad(outputs=loss, inputs=noisy_slice)[0]
                gamma_ = gamma / loss.item() 
                noisy_slice = noisy_slice - norm_grad * gamma_ 
                noisy_slice = scheduler.step(noise_pred, t, noisy_slice).prev_sample 
                new_noisy_volume[:, :, :, :, slice_3] = noisy_slice
            
            noisy_volume = new_noisy_volume.clone()
                
        else: 
            new_noisy_volume = noisy_volume.clone()
            
            with torch.no_grad():
                for slice_1 in range(volume_size):   
                    noisy_slice = noisy_volume[:, :, slice_1].clone()
                    noise_pred = model(noisy_slice, timesteps).sample
                    new_noisy_volume[:, :, slice_1] = scheduler.step(noise_pred, t, noisy_slice).prev_sample

            noisy_volume = new_noisy_volume.clone()

    return noisy_volume

num_inference_steps = 1000
volume_size = 128
recovered_volume_size = (128, 128, 128)
gamma = args.gamma
K = 3
interpolate = args.factor
slice1_ = [0]
slice2_ = [1]
slice3_ = [2]

noised_volume = np.load(path)

noised_volume_recon = np.zeros(recovered_volume_size)
indices = torch.arange(0, volume_size, interpolate)
noised_volume_recon[indices, :, :] = noised_volume 

noised_volume = torch.from_numpy(noised_volume).unsqueeze(0)
noised_volume = noised_volume / 255 * 2 - 1

generated_volumes = posterior_sample(model, noise_scheduler, noised_volume, num_inference_steps, volume_size, gamma, K, interpolate, slice1_, slice2_, slice3_)
generated_volumes = generated_volumes.detach().cpu().numpy().squeeze() 

np.save(output_path + '/output.npy', (generated_volumes + 1) / 2 * 255)

