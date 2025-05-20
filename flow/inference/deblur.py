import argparse
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchcfm.models.unet.unet import UNetModelWrapper
import tifffile

# Initialize device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gaussian_kernel(kernel_size, sigma, device):
    """Create a 2D Gaussian kernel"""
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            val = -(x**2 + y**2) / (2 * sigma**2)
            kernel[i, j] = torch.exp(torch.tensor(val, device=device))
    kernel = kernel / kernel.sum()  # Normalize
    return kernel.view(1, 1, kernel_size, kernel_size)

def load_model(model_path):
    """Load the pre-trained model from the specified path"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetModelWrapper(
        dim=(1, 128, 128),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    )
    
    # Load checkpoint and handle state dict keys
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt['ema_model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k.replace("module.", "")
        elif k.startswith("ema_"):
            new_key = k.replace("ema_", "")
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(True)
    return model, device

def posterior_sample_deblur2D(model, blurred_image, alpha, factor=1.0, num_inference_steps=100):
    """Perform deblurring using patch-based posterior sampling"""
    patch_size = 128  # Patch size
    stride = 112      # Stride for patch sliding
    
    # Target dimensions are same as blurred image
    original_height = blurred_image.shape[-2]
    original_width = blurred_image.shape[-1]
    
    # Initialize current estimate with random noise
    current_estimate = torch.randn(1, blurred_image.shape[0], original_height, original_width, device=device)
    
    # Create Gaussian blur kernel based on factor
    sigma = factor  # factor controls the standard deviation of the Gaussian
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Kernel size covers 3*sigma
    blur_kernel = gaussian_kernel(kernel_size, sigma, device)
    padding = kernel_size // 2
    
    # Generate patch coordinates
    patches_info = []
    for i in range((original_height - patch_size) // stride + 1 + (1 if (original_height - patch_size) % stride != 0 else 0)):
        for j in range((original_width - patch_size) // stride + 1 + (1 if (original_width - patch_size) % stride != 0 else 0)):
            start_y = i * stride
            start_x = j * stride
            if start_y + patch_size > original_height:
                start_y = original_height - patch_size
            if start_x + patch_size > original_width:
                start_x = original_width - patch_size
            patches_info.append((start_y, start_x))
    
    # Iterative sampling process
    for step in range(num_inference_steps):
        print(f"step {step + 1}")
        t = torch.tensor([step / (num_inference_steps - 1)], device=device)
        
        # Extract all patches from current estimate
        patches = []
        for (sy, sx) in patches_info:
            patch = current_estimate[0, :, sy:sy+patch_size, sx:sx+patch_size]
            patches.append(patch)
        patches = torch.stack(patches)
        
        # Process each patch
        for idx in range(len(patches)):
            # Current estimated sharp patch
            hr_patch = patches[idx].unsqueeze(0)
            
            # Corresponding blurred patch
            sy, sx = patches_info[idx]
            lr_patch = blurred_image[:, sy:sy+patch_size, sx:sx+patch_size].unsqueeze(0)
            
            # Calculate data fidelity gradient
            blurred_estimate = F.conv2d(hr_patch, blur_kernel, padding=padding)
            residual = blurred_estimate - lr_patch
            grad_data = F.conv_transpose2d(residual, blur_kernel, padding=padding)
            
            # Update step
            lr = (1 - t) ** alpha
            z = hr_patch - lr * grad_data
            
            # Add noise
            noise = torch.randn_like(z)
            z_tilde = t * z + (1 - t) * noise
            
            # Denoising step
            with torch.no_grad():
                model_output = model(t.expand(z_tilde.shape[0]), z_tilde)
                denoised = z_tilde + (1 - t) * model_output
            
            patches[idx] = denoised.squeeze(0)
        
        # Reconstruct full image
        reconstructed = torch.zeros_like(current_estimate)
        weight_mask = torch.zeros_like(current_estimate)
        
        for idx, (sy, sx) in enumerate(patches_info):
            reconstructed[..., sy:sy+patch_size, sx:sx+patch_size] += patches[idx]
            weight_mask[..., sy:sy+patch_size, sx:sx+patch_size] += 1
        
        current_estimate = reconstructed / (weight_mask + 1e-6)
    
    return current_estimate.squeeze(0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image deblurring using diffusion model')
    parser.add_argument('--input', type=str, required=True, help='Path to input blurred image')
    parser.add_argument('--output', type=str, default='results/deblur/', help='Path to save output deblurred image')
    parser.add_argument('--model_path', type=str, default='model/flow_weights.pt', help='Path to the pretrained model weights')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter for sampling (default: 0.3)')
    parser.add_argument('--factor', type=float, default=1.0, help='Factor for Gaussian blur kernel (default: 1.0)')
    parser.add_argument('--steps', type=int, default=100, help='Number of inference steps (default: 100)')
    
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, device = load_model(args.model_path)

    # Load and preprocess input image
    blurred_image = tifffile.imread(args.input)
    blurred_tensor = torch.from_numpy(blurred_image).float() * 2 / 255.0 - 1
    blurred_tensor = blurred_tensor.unsqueeze(0)  # Add channel dimension
    blurred_tensor = blurred_tensor.to(device)

    # Perform deblurring
    print("Starting deblurring process...")
    deblur_output = posterior_sample_deblur2D(
        model=model,
        blurred_image=blurred_tensor,
        alpha=args.alpha,
        factor=args.factor,
        num_inference_steps=args.steps,
    )
    
    # Convert output to image
    deblur_output = deblur_output.squeeze().cpu().numpy()
    deblur_output = ((deblur_output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    deblur_image = Image.fromarray(deblur_output)
    
    # Save output
    output_dir = args.output 
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.png")
    deblur_image.save(output_path)
    print(f"Deblurring completed. Result saved to {args.output}")

if __name__ == "__main__":
    main()