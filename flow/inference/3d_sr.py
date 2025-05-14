import argparse
import torch
import numpy as np
from PIL import Image
from torchcfm.models.unet.unet import UNetModelWrapper
import torch.nn.functional as F
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path

def downsample(x, factor):
    """Downsample along H-axis using bilinear interpolation."""
    B, C, H, W = x.shape
    new_H = H // factor
    return F.interpolate(x, size=(new_H, W), mode='bilinear', align_corners=False)

def upsample(y, factor):
    """Upsample H-axis to fixed size 128 using bilinear interpolation."""
    B, C, H, W = y.shape
    new_H = 128
    return F.interpolate(y, size=(new_H, W), mode='bilinear', align_corners=False)

def posterior_sample_sr3D(model, noised_volume, alpha, factor):
    """
    Perform 3D super-resolution via posterior sampling.

    Args:
        model: Pretrained diffusion model.
        noised_volume: Low-res input (tensor).
        alpha: Learning rate decay.
        factor: Upscaling factor.

    Returns:
        x0: Denoised and super-resolved volume.
    """
    num_steps = 100
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y0 = noised_volume.unsqueeze(0).to(device)
    x0 = torch.randn(1, 128, 128, 128).to(device)
    
    # Prepare batch indices for slicing
    batch_indices_list = [list(range(start, min(start + batch_size, x0.shape[1]))) 
                          for start in range(0, x0.shape[1], batch_size)]

    for step in range(num_steps):
        print(f"step {step + 1}")
        t = torch.tensor(step / num_steps, device=device, dtype=torch.float32)
        lr = (1 - t) ** alpha
        mod = step % 3

        for batch_idx, batch_indices in enumerate(batch_indices_list):
            current_batch_size = len(batch_indices)
            with torch.no_grad():
                if mod == 0:
                    x_batch = x0[:, batch_indices, :, :].permute(1, 0, 2, 3)
                elif mod == 1:
                    x_batch = x0[:, :, batch_indices, :].permute(2, 0, 1, 3)
                    y_batch = y0[:, :, batch_indices, :].permute(2, 0, 1, 3)
                else:
                    x_batch = x0[:, :, :, batch_indices].permute(3, 0, 1, 2)
                    y_batch = y0[:, :, :, batch_indices].permute(3, 0, 1, 2)

                if mod == 0:
                    z_batch = x_batch
                else:
                    Hz_batch = downsample(x_batch, factor)
                    z_batch = x_batch - lr * upsample(Hz_batch - y_batch, factor)

                zt_batch = t * z_batch + torch.randn_like(z_batch) * (1 - t)
                v_batch = model(t.expand(zt_batch.shape[0]), zt_batch)
                x_update_batch = zt_batch + (1 - t) * v_batch

                if mod == 0:
                    x0[:, batch_indices, :, :] = x_update_batch.permute(1, 0, 2, 3)
                elif mod == 1:
                    x0[:, :, batch_indices, :] = x_update_batch.permute(1, 2, 0, 3)
                else:
                    x0[:, :, :, batch_indices] = x_update_batch.permute(1, 2, 3, 0)

                del x_batch, z_batch, zt_batch, v_batch, x_update_batch
                torch.cuda.empty_cache()

    return x0

def main():
    parser = argparse.ArgumentParser(description='3D Super Resolution using Diffusion Models')
    parser.add_argument('--input', type=str, required=True, help='Path to input volume (.mrc or .npy)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--factor', type=int, required=True, help='Super resolution factor')
    parser.add_argument('--alpha', type=float, default=0.3, help='Learning rate decay')
    parser.add_argument('--model_path', type=str, default="/root/otcfm_weights_step_18000.pt", help='Model weights path')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    if args.input.endswith('.mrc'):
        with mrcfile.open(args.input) as mrc:
            noised_image = mrc.data
    elif args.input.endswith('.npy'):
        noised_image = np.load(args.input)
    else:
        raise ValueError("Input must be MRC or NPY format")
    
    # Normalize and convert to tensor
    noised_image = torch.from_numpy(noised_image.astype(np.float32) / 255 * 2 - 1)
    
    # Load model
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
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    state_dict = ckpt['ema_model']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "").replace("ema_", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(True)

    # Crop input
    noised_image = noised_image.squeeze()
    factor = args.factor
    s = 128 // factor 

    D, H, W = noised_image.shape
    D_blocks = D // s
    H_blocks = H // 128
    W_blocks = W // 128

    D_valid = D_blocks * s
    H_valid = H_blocks * 128
    W_valid = W_blocks * 128

    noised_image_cropped = noised_image[:D_valid, :H_valid, :W_valid]

    output_D = D_blocks * (128 // factor * factor) 
    output = np.zeros((output_D, H_valid, W_valid), dtype=np.float32)

    # Process blocks
    for i in range(D_blocks):
        for j in range(H_blocks):
            for m in range(W_blocks):
                if 128 % factor != 0:
                    input_block = noised_image_cropped[i*s : (i+1)*s + 1, j*128 : (j+1)*128, m*128 : (m+1)*128]
                else:
                    input_block = noised_image_cropped[i*s : (i+1)*s, j*128 : (j+1)*128, m*128 : (m+1)*128]

                generated_volumes = posterior_sample_sr3D(model, input_block, args.alpha, factor)
                generated_volumes = generated_volumes.detach().cpu().numpy().squeeze()
                generated_volumes = (generated_volumes + 1) / 2 * 255

                output[i*(128 // factor * factor): (i+1)*(128 // factor * factor),
                       j*128 : (j+1)*128,
                       m*128 : (m+1)*128] = generated_volumes[:(128 // factor * factor)]

    # Save results
    np.save(output_dir / "output.npy", output)
    with mrcfile.new(output_dir / "output.mrc", overwrite=True) as mrc:
        mrc.set_data(output)

    plt.imsave(output_dir / "output.png", output[1, :, :], cmap='gray', vmin=0, vmax=255)
    plt.imsave(output_dir / "input.png", (noised_image[1, :, :] + 1) / 2 * 255, cmap='gray', vmin=0, vmax=255)

if __name__ == '__main__':
    main()