import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import argparse

# set some parameters
parser = argparse.ArgumentParser(description='DPS denoising')
parser.add_argument('--path', type=str, default='', help='input data path')
parser.add_argument('--gamma', type=float, default=1.0, help='step size')

args = parser.parse_args()
path = args.path
gamma = args.gamma

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


def posterior_sample(model, scheduler, noised_image, num_inference_steps=1000, batch_size=1, gamma=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_size = 256
    stride = 224
    original_height = noised_image.shape[-2]
    original_width = noised_image.shape[-1]

    # Convert to torch tensor
    noised_image_large = noised_image.to(device)

    # Calculate the number of patches along each dimension
    num_patches_y = (original_height - patch_size) // stride + 1
    num_patches_x = (original_width - patch_size) // stride + 1

    # Add extra patches to cover the boundaries if needed
    if (original_height - patch_size) % stride != 0:
        num_patches_y += 1
    if (original_width - patch_size) % stride != 0:
        num_patches_x += 1

    # Extract patches, including boundary handling
    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            start_y = i * stride
            start_x = j * stride

            # Handle boundaries by adjusting the start position
            if start_y + patch_size > original_height:
                start_y = original_height - patch_size
            if start_x + patch_size > original_width:
                start_x = original_width - patch_size

            patch = noised_image_large[:, start_y:start_y + patch_size, start_x:start_x + patch_size]
            patches.append(patch)

    # Stack patches into a tensor
    noised_image = torch.stack(patches)
    print("small images:", noised_image.shape)

    # Initialize a noisy image
    noisy_image_large = torch.randn((batch_size, model.config.in_channels, noised_image_large.shape[1], noised_image_large.shape[2]), device=device, requires_grad=True)
    
    # Inference loop
    for t in reversed(range(num_inference_steps)):

        # crop in to small patches with overlap 
        patches = []
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride
                start_x = j * stride

                # Handle boundaries by adjusting the start position
                if start_y + patch_size > original_height:
                    start_y = original_height - patch_size
                if start_x + patch_size > original_width:
                    start_x = original_width - patch_size

                patch = noisy_image_large[0, 0, start_y:start_y + patch_size, start_x:start_x + patch_size]
                patches.append(patch)
        
        noisy_image = torch.stack(patches)
        noisy_image = noisy_image.unsqueeze(1)
        noisy_image = noisy_image.clone()

        minibatch_size = 8

        num_minibatches = (noisy_image.shape[0] + minibatch_size - 1) // minibatch_size

        for i in range(num_minibatches):
            start_idx = i * minibatch_size
            end_idx = min((i + 1) * minibatch_size, noisy_image.shape[0])

            noisy_batch = noisy_image[start_idx:end_idx]

            timesteps = torch.full((noisy_batch.shape[0],), t, dtype=torch.long, device=device)

            with torch.enable_grad():
                noise_pred = model(noisy_batch, timesteps).sample
                alpha_t = scheduler.alphas_cumprod[t]
                noisy_image_0 = (noisy_batch - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            loss = torch.zeros(1, requires_grad=True, device=device)

            difference = noised_image[start_idx:end_idx] - noisy_image_0
            loss = loss + torch.linalg.norm(difference)

            norm_grad = torch.autograd.grad(outputs=loss, inputs=noisy_batch)[0]

            gamma_ = gamma / loss.item()
            noisy_batch = noisy_batch - norm_grad * gamma_

            noisy_batch = scheduler.step(noise_pred, t, noisy_batch).prev_sample

            noisy_image[start_idx:end_idx] = noisy_batch
                
        print(f"step: {t}", f"loss: {loss.item():.4f}", f"gamma: {gamma_:.4f}")

        # recover to large image 
        reconstructed = torch.zeros(1, model.config.in_channels, original_height, original_width, device=noisy_image.device)
        weight_mask = torch.zeros_like(reconstructed)

        patch_idx = 0
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride
                start_x = j * stride

                # Handle boundaries by adjusting the start position
                if start_y + patch_size > original_height:
                    start_y = original_height - patch_size
                if start_x + patch_size > original_width:
                    start_x = original_width - patch_size

                patch = noisy_image[patch_idx]
                patch_idx += 1

                reconstructed[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] += patch
                weight_mask[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] += 1

        reconstructed /= weight_mask
        noisy_image_large = reconstructed

    return noisy_image_large, noisy_image

num_inference_steps = 1000
batch_size = 1
gamma = gamma

noised_image = np.load(path)
noised_image = noised_image 
plt.imsave(f"{path[:-4]}.png", noised_image, cmap='gray', vmin=0, vmax=255) 

noised_image = torch.from_numpy(noised_image).unsqueeze(0)
noised_image = noised_image / 255 * 2 - 1

generated_image, generated_patches = posterior_sample(model, noise_scheduler, noised_image, num_inference_steps, batch_size, gamma)
generated_image = generated_image.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
plt.imsave(f"{path[:-4]}_{checkpoint}_{gamma}_in_distribution.png", (generated_image + 1) / 2 * 255, cmap='gray', vmin=0, vmax=255) 
