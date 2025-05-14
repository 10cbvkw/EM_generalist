import argparse
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchcfm.models.unet.unet import UNetModelWrapper

# Initialize device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def posterior_sample_sr2D(model, low_res_image, alpha, factor, num_inference_steps=100):
    """Perform super-resolution using patch-based posterior sampling"""
    patch_size = 128  # High-resolution patch size
    stride = 112      # Stride for patch sliding
    
    # Calculate target high-resolution dimensions
    original_lr_height = low_res_image.shape[-2]
    original_lr_width = low_res_image.shape[-1]
    target_height = original_lr_height * factor
    target_width = original_lr_width * factor
    
    # Initialize high-resolution image
    current_estimate = torch.randn(1, low_res_image.shape[0], target_height, target_width, device=device)
    
    # Calculate number of patches needed
    num_patches_y = (target_height - patch_size) // stride + 1
    num_patches_x = (target_width - patch_size) // stride + 1
    num_patches_y += 1 if (target_height - patch_size) % stride != 0 else 0
    num_patches_x += 1 if (target_width - patch_size) % stride != 0 else 0
    
    # Generate mapping between low-res and high-res patches
    lr_patch_size = patch_size // factor
    lr_stride = stride // factor
    
    hr_patches_info = []
    lr_patches_info = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # High-res patch coordinates
            start_y = i * stride
            start_x = j * stride
            if start_y + patch_size > target_height:
                start_y = target_height - patch_size
            if start_x + patch_size > target_width:
                start_x = target_width - patch_size
            hr_patches_info.append((start_y, start_x))
            
            # Corresponding low-res patch coordinates
            lr_start_y = start_y // factor
            lr_start_x = start_x // factor
            if lr_start_y + lr_patch_size > original_lr_height:
                lr_start_y = original_lr_height - lr_patch_size
            if lr_start_x + lr_patch_size > original_lr_width:
                lr_start_x = original_lr_width - lr_patch_size
            lr_patches_info.append((lr_start_y, lr_start_x))
    
    # Iterative sampling process
    for step in range(num_inference_steps):
        t = torch.tensor([step / (num_inference_steps - 1)], device=device)
        
        # Extract all patches from current estimate
        patches = []
        for (sy, sx) in hr_patches_info:
            patch = current_estimate[0, :, sy:sy+patch_size, sx:sx+patch_size]
            patches.append(patch)
        patches = torch.stack(patches)
        
        # Process each patch
        for idx in range(len(patches)):
            # Current high-res patch
            hr_patch = patches[idx].unsqueeze(0)
            
            # Corresponding low-res patch
            lr_sy, lr_sx = lr_patches_info[idx]
            lr_patch = low_res_image[:, lr_sy:lr_sy+lr_patch_size, lr_sx:lr_sx+lr_patch_size].unsqueeze(0)
            
            # Calculate data fidelity gradient (guided by downsampling difference)
            downsampled = F.interpolate(hr_patch, size=(lr_patch_size, lr_patch_size), mode='bilinear', align_corners=False)
            grad_data = F.interpolate(downsampled - lr_patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            
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
        
        for idx, (sy, sx) in enumerate(hr_patches_info):
            reconstructed[..., sy:sy+patch_size, sx:sx+patch_size] += patches[idx]
            weight_mask[..., sy:sy+patch_size, sx:sx+patch_size] += 1
        
        current_estimate = reconstructed / (weight_mask + 1e-6)  # Avoid division by zero
    
    return current_estimate.squeeze(0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Super-resolution using diffusion model')
    parser.add_argument('--input', type=str, required=True, help='Path to input low-resolution image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output high-resolution image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model weights')
    parser.add_argument('--factor', type=int, default=4, help='Super-resolution factor (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter for sampling (default: 0.3)')
    parser.add_argument('--steps', type=int, default=100, help='Number of inference steps (default: 100)')
    
    args = parser.parse_args()

    # Load model
    try:
        print("Loading model...")
        model, device = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess input image
    try:
        lr_image = Image.open(args.input).convert('L')  # Convert to grayscale
        lr_tensor = torch.from_numpy(np.array(lr_image)).float() * 2 / 255.0 - 1
        lr_tensor = lr_tensor.unsqueeze(0)  # Add channel dimension
        lr_tensor = lr_tensor.to(device)
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    # Perform super-resolution
    try:
        print("Starting super-resolution process...")
        hr_output = posterior_sample_sr2D(
            model=model,
            low_res_image=lr_tensor,
            alpha=args.alpha,
            factor=args.factor,
            num_inference_steps=args.steps,
            batch_size=args.batch_size
        )
        
        # Convert output to image
        hr_output = hr_output.squeeze().cpu().numpy()
        hr_output = ((hr_output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        hr_image = Image.fromarray(hr_output)
        
        # Save output
        hr_image.save(args.output)
        print(f"Super-resolution completed. Result saved to {args.output}")
        
    except Exception as e:
        print(f"Error during super-resolution: {e}")

if __name__ == "__main__":
    main()