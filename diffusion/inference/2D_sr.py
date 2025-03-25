import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import argparse

# set some parameters
parser = argparse.ArgumentParser(description='DPS 2D suepr-resolution')
parser.add_argument('--path', type=str, default='', help='input path')
parser.add_argument('--gamma', type=float, default=1.0, help='step size')
parser.add_argument('--factor', type=int, default=2, help='fold factor for super-resolution')

args = parser.parse_args()
path = args.path
gamma = args.gamma
factor = args.factor

epoch = 999
checkpoint = ''

# Load the model and scheduler
model_dir = f"../train/exp/{checkpoint}/weights_ema.epoch_{epoch}.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_dir).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# Set model to evaluation mode
model.eval()

# Ensure gradients are retained for the generated images
torch.set_grad_enabled(True)


def posterior_sample(model, scheduler, noised_image, num_inference_steps=1000, batch_size=1, gamma=1, factor=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_size_small = 128 // factor
    if (128 % factor) != 0:
        patch_size_small = patch_size_small + 1
    stride_small = 112 // factor
    if (112 % factor) != 0:
        stride_small = stride_small + 1
    patch_size_large = 128
    stride_large = 112 

    original_height_small = noised_image.shape[-2]
    original_width_small = noised_image.shape[-1]
    original_height_large = noised_image.shape[-2] * factor
    original_width_large = noised_image.shape[-1] * factor

    # Convert to torch tensor
    noised_image_large = noised_image.to(device)

    # Calculate the number of patches along each dimension
    num_patches_y = (original_height_small - patch_size_small) // stride_small + 1
    num_patches_x = (original_width_small - patch_size_small) // stride_small + 1

    # Add extra patches to cover the boundaries if needed
    if (original_height_small - patch_size_small) % stride_small != 0:
        num_patches_y += 1
    if (original_width_small - patch_size_small) % stride_small != 0:
        num_patches_x += 1

    # Extract patches, including boundary handling
    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            start_y = i * stride_small
            start_x = j * stride_small

            # Handle boundaries by adjusting the start position
            if start_y + patch_size_small > original_height_small:
                start_y = original_height_small - patch_size_small
            if start_x + patch_size_small > original_width_small:
                start_x = original_width_small - patch_size_small

            patch = noised_image_large[:, start_y:start_y + patch_size_small, start_x:start_x + patch_size_small]
            patches.append(patch)

    # Stack patches into a tensor
    noised_image = torch.stack(patches)
    print("small images:", noised_image.shape)

    # Initialize a noisy image
    noisy_image_large = torch.randn((batch_size, model.config.in_channels, original_height_large, original_width_large), device=device, requires_grad=True)
    
    # Inference loop
    for t in reversed(range(num_inference_steps)):

        # crop in to small patches with overlap 
        patches = []
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride_large
                start_x = j * stride_large

                # Handle boundaries by adjusting the start position
                if start_y + patch_size_large > original_height_large:
                    start_y = original_height_large - patch_size_large
                if start_x + patch_size_large > original_width_large:
                    start_x = original_width_large - patch_size_large

                patch = noisy_image_large[0, 0, start_y:start_y + patch_size_large, start_x:start_x + patch_size_large]
                patches.append(patch)

        noisy_image = torch.stack(patches)
        noisy_image = noisy_image.unsqueeze(1)
        noisy_image = noisy_image.clone()

        for k in range(noisy_image.shape[0]):

            # Get the current timestep tensor
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            noisy_patch = noisy_image[k].unsqueeze(0)

            # Predict the noise residual
            with torch.enable_grad():
                
                noise_pred = model(noisy_patch, timesteps).sample
                alpha_t = scheduler.alphas_cumprod[t]
                noisy_image_0 = (noisy_patch - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            loss = torch.zeros(1, requires_grad=True, device=device)

            noisy_image_0_small = noisy_image_0[:, :, ::factor, ::factor]
            
            difference = noised_image[k] - noisy_image_0_small 

            loss = loss + torch.linalg.norm(difference) 

            norm_grad = torch.autograd.grad(outputs=loss, inputs=noisy_patch)[0]
            
            gamma_ = gamma / loss.item() 
            noisy_patch = noisy_patch - norm_grad * gamma_ 
            
            noisy_image[k] = noisy_patch
            noisy_image[k] = scheduler.step(noise_pred, t, noisy_image[k]).prev_sample
        
        print(f"step: {t}", f"loss: {loss.item():.4f}", f"gamma: {gamma_:.4f}")

        # recover to large image 
        reconstructed = torch.zeros(1, model.config.in_channels, original_height_large, original_width_large, device=noisy_image.device)
        weight_mask = torch.zeros_like(reconstructed)

        patch_idx = 0
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride_large
                start_x = j * stride_large

                # Handle boundaries by adjusting the start position
                if start_y + patch_size_large > original_height_large:
                    start_y = original_height_large - patch_size_large
                if start_x + patch_size_large > original_width_large:
                    start_x = original_width_large - patch_size_large

                patch = noisy_image[patch_idx]
                patch_idx += 1

                reconstructed[:, :, start_y:start_y + patch_size_large, start_x:start_x + patch_size_large] += patch
                weight_mask[:, :, start_y:start_y + patch_size_large, start_x:start_x + patch_size_large] += 1

        reconstructed /= weight_mask
        noisy_image_large = reconstructed

    return noisy_image_large, noisy_image

num_inference_steps = 1000
batch_size = 1

noised_image = np.load(path)

noised_image = torch.from_numpy(noised_image).unsqueeze(0)
noised_image = noised_image / 255 * 2 - 1
print(noised_image.shape, noised_image.max(), noised_image.min())

generated_image, generated_patches = posterior_sample(model, noise_scheduler, noised_image, num_inference_steps, batch_size, gamma, factor)
generated_image = generated_image.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
generated_patches = generated_patches.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()

plt.imsave(f"{path[:-4]}_{checkpoint}_{gamma}_{factor}.png", (generated_image + 1) / 2 * 255, cmap='gray', vmin=0, vmax=255) 
