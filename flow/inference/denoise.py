import argparse
import torch
import numpy as np
from PIL import Image
from torchcfm.models.unet.unet import UNetModelWrapper

# Initialize device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load and initialize the UNet model with given weights"""
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
    
    # Load checkpoint and adjust state dict keys
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
    return model

def posterior_sample_denoise(model, noised_image, alpha, num_inference_steps=100):
    """Perform patch-based denoising using the model"""
    patch_size = 128
    stride = 112
    original_height = noised_image.shape[-2]
    original_width = noised_image.shape[-1]
    noised_image_large = noised_image.to(device)
    
    # Calculate number of patches needed
    num_patches_y = (original_height - patch_size) // stride + 1
    num_patches_x = (original_width - patch_size) // stride + 1
    if (original_height - patch_size) % stride != 0:
        num_patches_y += 1
    if (original_width - patch_size) % stride != 0:
        num_patches_x += 1
    
    # Extract patches from the noisy image
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
    
    noised_image = torch.stack(patches)  # Initial noisy patches
    noisy_image_large = torch.ones_like(noised_image_large, device=device)  # Initialize with ones
    noisy_image_large = noisy_image_large.unsqueeze(0)

    # Denoising loop
    for step in range(num_inference_steps):
        t = step / (num_inference_steps - 1)  # t ranges from 0 to 1
        t = torch.full((1,), t, device=device)

        # Extract current patches from the large image
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
        noisy_image = torch.stack(patches).unsqueeze(1).clone()

        # Process each patch
        for k in range(noisy_image.shape[0]):
            x = noisy_image[k].unsqueeze(0)  
            y = noised_image[k].unsqueeze(0)  
            lr = (1 - t) ** alpha
            grad = x - y  # Data fidelity gradient
            z = x - lr.view(-1, 1, 1, 1) * grad
            noise = torch.randn_like(z)
            z_tilde = t.view(-1, 1, 1, 1) * z + (1 - t.view(-1, 1, 1, 1)) * noise  # Interpolation step
            
            with torch.no_grad():
                model_output = model(t.expand(z_tilde.shape[0]), z_tilde)
                denoised = z_tilde + (1 - t.view(-1, 1, 1, 1)) * model_output  # Denoising step
            noisy_image[k] = denoised.squeeze(0)

        # Reconstruct the full image from patches
        reconstructed = torch.zeros(1, noised_image_large.shape[0], original_height, original_width, device=device)
        weight_mask = torch.zeros_like(reconstructed)
        patch_idx = 0
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_y = i * stride
                start_x = j * stride
                if start_y + patch_size > original_height:
                    start_y = original_height - patch_size
                if start_x + patch_size > original_width:
                    start_x = original_width - patch_size
                patch = noisy_image[patch_idx]
                patch_idx += 1
                reconstructed[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] += patch
                weight_mask[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] += 1
        
        noisy_image_large = reconstructed / weight_mask
    
    return noisy_image_large

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image denoising inference')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output image')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Data fidelity weight coefficient (default: 0.3)')
    parser.add_argument('--num_inference_steps', type=int, default=100,
                       help='Number of inference steps (default: 100)')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Load and preprocess input image
    try:
        input_image = Image.open(args.input).convert('L')  # Convert to grayscale
    except Exception as e:
        print(f"Failed to load input image: {e}")
        return

    # Convert to PyTorch tensor and normalize
    image_np = np.array(input_image)
    image_tensor = torch.tensor(image_np, dtype=torch.float32) * 2 / 255.0 - 1
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add channel dimension [1, H, W]

    # Perform denoising
    with torch.no_grad():
        denoised_tensor = posterior_sample_denoise(
            model=model,
            noised_image=image_tensor,
            alpha=args.alpha,
            num_inference_steps=args.num_inference_steps
        )

    # Post-process and save result
    denoised_np = denoised_tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    denoised_np = ((denoised_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(denoised_np).save(args.output)
    print(f"Processing complete. Result saved to: {args.output}")

if __name__ == "__main__":
    main()