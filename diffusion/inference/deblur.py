import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from scipy.optimize import minimize

from functools import partial
import yaml
import pickle

parser = argparse.ArgumentParser(description='DPS deblurring')
parser.add_argument('--gamma', type=float, default=5, help='step size')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for kernel update')
parser.add_argument('--max_iter', type=int, default=10, help='max iteration for kernel update')
parser.add_argument('--kernel_size', type=int, default=33, help='kernel size')
parser.add_argument('--path', type=str, default='', help='input path')
parser.add_argument('--center_penalty', type=float, default=0.0, help='center_penalty weight')
parser.add_argument('--l1_penalty', type=float, default=0.0, help='l1_penalty weight')
parser.add_argument('--l2_penalty', type=float, default=0.0, help='l2_penalty weight')
parser.add_argument('--sigma_center', type=float, default=7.5, help='size of gaussian kernel for center penalty')
parser.add_argument('--output_path', type=str, help='output path')

args = parser.parse_args()

gamma = args.gamma
kernel_size = args.kernel_size
lr = args.lr
max_iter = args.max_iter
input_path = args.path
output_path = args.output_path
center_penalty = args.center_penalty
l1_penalty = args.l1_penalty
l2_penalty = args.l2_penalty
sigma_center = args.sigma_center

# Load the model and scheduler
model_dir = f"../train/exp/prior_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_dir).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# Create kernel
def create_kernel(device, size=13):

    k_start = torch.zeros((1, 1, size, size)).to(device)
    k_start[0, 0, size//2, size//2] = 1

    return k_start

def update_kernel(blurred_image, estimated_sharp_image, initial_kernel, lr_=0.01, max_iter_=20, device='cuda',
                  l1_penalty=0.0, l2_penalty=0.0, center_penalty=0.0, sigma_center=5.0):  

    blurred_image_data = blurred_image.detach().to(device)
    estimated_sharp_image_data = estimated_sharp_image.detach().to(device)
    initial_kernel_data = initial_kernel.detach().to(device)

    kernel = torch.nn.Parameter(initial_kernel_data)
    kernel_size = kernel.shape[-1]
    optimizer_m_step = torch.optim.LBFGS([kernel], lr=lr_, max_iter=max_iter_)

    def closure():
        optimizer_m_step.zero_grad()
        reblurred_image = F.conv2d(estimated_sharp_image_data, kernel)
        
        output_height_start = kernel_size // 2
        output_width_start = kernel_size // 2
        output_height_end = blurred_image_data.shape[-2] - kernel_size // 2 
        output_width_end = blurred_image_data.shape[-1] - kernel_size // 2

        blurred_image_data_cropped = blurred_image_data[:, output_height_start:output_height_end, 
                                                        output_width_start:output_width_end]

        loss_m_step = F.mse_loss(reblurred_image, blurred_image_data_cropped.unsqueeze(1))
        
        l1_reg = l1_penalty * torch.norm(kernel, p=1)
        l2_reg = l2_penalty * torch.sum(kernel ** 2)
        
        kernel_2d = kernel.squeeze()
        y_indices = torch.arange(kernel_size, device=device).float()
        x_indices = torch.arange(kernel_size, device=device).float()
        y, x = torch.meshgrid(y_indices, x_indices, indexing='ij')
        center_g = (kernel_size - 1) / 2.0
        
        sigma = sigma_center  
        gaussian = torch.exp(-((x - center_g)**2 + (y - center_g)**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()   
        
        kernel_non_neg = kernel_2d.clamp(min=0)
        kernel_sum = kernel_non_neg.sum()
        eps = 1e-10
        if kernel_sum < eps:
            kernel_normalized = torch.ones_like(kernel_non_neg) / (kernel_size**2)
        else:
            kernel_normalized = kernel_non_neg / kernel_sum
        
        cross_entropy = - torch.sum(kernel_normalized * torch.log(gaussian + eps)) + torch.sum(kernel_normalized * torch.log(kernel_normalized + eps))
        center_reg = center_penalty * cross_entropy
        
        total_loss = loss_m_step + l1_reg + l2_reg + center_reg
        total_loss.backward()
        return total_loss
    
    optimizer_m_step.step(closure)

    with torch.no_grad():
        kernel.data = kernel.data / kernel.data.sum()

    return kernel

model.eval()

torch.set_grad_enabled(True)


def posterior_sample(model, scheduler, noised_image, num_inference_steps=1000, batch_size=1, gamma_const=1, kernel_size=33, lr=0.01, max_iter=20, l1_penalty=0.0, l2_penalty=0.0, center_penalty=0.0, sigma_center=5.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_size = 128
    stride = 112
    original_height = noised_image.shape[-2]
    original_width = noised_image.shape[-1]

    noised_image_large = noised_image.to(device)

    num_patches_y = (original_height - patch_size) // stride + 1
    num_patches_x = (original_width - patch_size) // stride + 1

    if (original_height - patch_size) % stride != 0:
        num_patches_y += 1
    if (original_width - patch_size) % stride != 0:
        num_patches_x += 1

    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            start_y = i * stride
            start_x = j * stride

            if start_y + patch_size > original_height:
                start_y = original_height - patch_size
            if start_x + patch_size > original_width:
                start_x = original_width - patch_size

            patch = noised_image_large[:, start_y:start_y + patch_size, start_x:start_x + patch_size]
            patches.append(patch)

    noised_image = torch.stack(patches)
    print("small images:", noised_image.shape)

    noisy_image_large = torch.randn((batch_size, model.config.in_channels, noised_image_large.shape[1], noised_image_large.shape[2]), device=device, requires_grad=True)
    kernel = create_kernel(device, kernel_size)

    for t in reversed(range(num_inference_steps)):
        
        gamma = gamma_const # * (10 - 9 * (1 - t / num_inference_steps))

        patches = []
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride
                start_x = j * stride

                if start_y + patch_size > original_height:
                    start_y = original_height - patch_size
                if start_x + patch_size > original_width:
                    start_x = original_width - patch_size

                patch = noisy_image_large[0, 0, start_y:start_y + patch_size, start_x:start_x + patch_size]
                patches.append(patch)
        noisy_image = torch.stack(patches)
        noisy_image = noisy_image.unsqueeze(1)
        noisy_image = noisy_image.clone()

        for k in range(noisy_image.shape[0]):

            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            noisy_patch = noisy_image[k].unsqueeze(0)

            with torch.enable_grad():
                
                noise_pred = model(noisy_patch, timesteps).sample
                alpha_t = scheduler.alphas_cumprod[t]
                noisy_image_0 = (noisy_patch - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            kernel_size = kernel.shape[-1]  
            
            output_height_start = kernel_size // 2
            output_width_start = kernel_size // 2
            
            output_height_end = noised_image[k].shape[-2] - kernel_size // 2 
            output_width_end = noised_image[k].shape[-1] - kernel_size // 2 

            noised_image_cropped = noised_image[k][:, output_height_start:output_height_end, output_width_start:output_width_end]
            noisy_image_0 = F.conv2d(noisy_image_0, kernel)

            difference = noised_image_cropped - noisy_image_0
            
            loss = torch.zeros(1, requires_grad=True, device=device)
            loss = loss + torch.linalg.norm(difference)

            norm_grad = torch.autograd.grad(outputs=loss, inputs=noisy_patch)[0]
            
            gamma_ = gamma / loss.item() 
            
            if t != 0:
                noisy_patch = noisy_patch - norm_grad * gamma_ 
            
            noisy_image[k] = noisy_patch
            noisy_image[k] = scheduler.step(noise_pred, t, noisy_image[k]).prev_sample
        
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

        noised_image_large_copy = noised_image_large.detach()
        noised_image_large_copy.requires_grad = False    
        noisy_image_large_copy = noisy_image_large.detach()
        noisy_image_large_copy.requires_grad = False

        kernel = update_kernel(noised_image_large_copy, noisy_image_large_copy, kernel, lr_=lr, max_iter_=max_iter, device=device, l1_penalty=l1_penalty, l2_penalty=l2_penalty, center_penalty=center_penalty, sigma_center=sigma_center)
    
    return noisy_image_large, noisy_image, kernel

num_inference_steps = 1000
batch_size = 1

noised_image = np.load(args.input_path)
print(noised_image.shape, noised_image.min(), noised_image.max())

plt.imsave(input_path[:-4]+f".png", noised_image, cmap='gray', vmin=0, vmax=255) 
    
noised_image = torch.from_numpy(noised_image).unsqueeze(0)
noised_image = noised_image / 255 * 2 - 1
print(noised_image.shape, noised_image.max(), noised_image.min())

generated_image, generated_patches, kernel = posterior_sample(model, noise_scheduler, noised_image, num_inference_steps, batch_size, gamma, kernel_size, lr, max_iter, l1_penalty, l2_penalty, center_penalty, sigma_center)
generated_image = generated_image.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
generated_patches = generated_patches.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
kernel = kernel.detach().cpu().numpy().squeeze()

print(generated_image.shape, generated_image.min(), generated_image.max())
plt.imsave(input_path[:-4]+f"{gamma}_{kernel_size}_deblur_{l1_penalty}_{l2_penalty}_{center_penalty}_{sigma_center}.png", (generated_image + 1) / 2 * 255, cmap='gray', vmin=0, vmax=255) 
plt.imsave(input_path[:-4]+f"{gamma}_{kernel_size}_deblur_{l1_penalty}_{l2_penalty}_{center_penalty}_{sigma_center}_kernel.png", kernel, cmap='gray', vmin=0, vmax=kernel.max())