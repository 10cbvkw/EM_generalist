import os
import copy
import torch
from absl import app, flags
from dataset import CTDataset, TifImageDataset
from tqdm import trange
from utils import ema, generate_samples, infiniteloop
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# Define all flags including the dataset path
flags.DEFINE_string("dataset_path", "data/EMDInverse_dataset", "Path to the dataset directory")
flags.DEFINE_string("output_dir", "./model/", "Output directory for saving models")
flags.DEFINE_integer("resume_step", 0, "Resume training from this step (0 means start from scratch)")
flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_integer("total_steps", 20001, "Total number of training steps")
flags.DEFINE_integer("warmup", 1000, "Number of warmup steps for learning rate")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_workers", 8, "Number of workers for DataLoader")
flags.DEFINE_float("ema_decay", 0.999, "EMA decay rate")
flags.DEFINE_integer("save_step", 1000, "Frequency of saving checkpoints (in steps)")
flags.DEFINE_bool("parallel", True, "Whether to use multi-GPU training")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def warmup_lr(step):
    """Linear warmup for learning rate until warmup steps, then full lr."""
    return 1.0 if step >= FLAGS.warmup else step / FLAGS.warmup

def load_checkpoint(step, net_model, ema_model, optimizer, scheduler):
    """Attempt to load checkpoint corresponding to given step from output_dir."""
    checkpoint_path = os.path.join(FLAGS.output_dir, f"otcfm_weights_step_{step}.pt")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        print(f"[INFO] Checkpoint loaded from {checkpoint_path}. Resuming from step {step}.")

        if step >= FLAGS.warmup:
            scheduler.base_lrs = [FLAGS.lr]
            for group in optimizer.param_groups:
                group['lr'] = FLAGS.lr
    else:
        if step > 0:
            print(f"[WARNING] Checkpoint '{checkpoint_path}' not found. Starting from scratch.")
        else:
            print("[INFO] Starting from scratch.")

def train(_):
    print(f"[INFO] Training configuration:")
    print(f"  - Dataset path: {FLAGS.dataset_path}")
    print(f"  - Learning rate: {FLAGS.lr}")
    print(f"  - Total steps: {FLAGS.total_steps}")
    print(f"  - EMA decay: {FLAGS.ema_decay}")
    print(f"  - Checkpoint save frequency: {FLAGS.save_step} steps")

    # 1. Prepare data using the FLAGS.dataset_path
    dataset = TifImageDataset(FLAGS.dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, 
        num_workers=FLAGS.num_workers, drop_last=True
    )
    datalooper = infiniteloop(dataloader)

    # 2. Initialize model
    net_model = UNetModelWrapper(
        dim=(1, 128, 128),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    ema_model = copy.deepcopy(net_model)

    # 3. Optimizer & scheduler
    optimizer = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_lr(step + FLAGS.resume_step)
    )

    # 4. Multi-GPU setup
    if FLAGS.parallel and torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel across {torch.cuda.device_count()} GPUs.")
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    total_params = sum(p.numel() for p in net_model.parameters()) / (1024 ** 2)
    print(f"[INFO] Model size: {total_params:.2f} MB parameters")

    # 5. Initialize Flow Matcher
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # 6. Create output directory & load checkpoint
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    load_checkpoint(FLAGS.resume_step, net_model, ema_model, optimizer, scheduler)

    # 7. Training loop
    with trange(FLAGS.resume_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # Get batch of data
            real_data = next(datalooper).to(device)
            noise_data = torch.randn_like(real_data)
            
            # Flow Matching
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(noise_data, real_data)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            ema(net_model, ema_model, FLAGS.ema_decay)

            # Periodic saving
            if FLAGS.save_step > 0 and (step % FLAGS.save_step == 0):
                generate_samples(net_model, FLAGS.parallel, FLAGS.output_dir, step, net="normal")
                generate_samples(ema_model, FLAGS.parallel, FLAGS.output_dir, step, net="ema")

                checkpoint = {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": scheduler.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step,
                }
                ckpt_path = os.path.join(FLAGS.output_dir, f"otcfm_weights_step_{step}.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"[INFO] Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    app.run(train)