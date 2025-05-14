import copy
import torch
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(model, parallel, save_dir, step, net="normal"):
    model.eval()
    if parallel:
        model = copy.deepcopy(model.module).to(device)
    node = NeuralODE(model, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        z = torch.randn(16, 1, 128, 128, device=device)
        traj = node.trajectory(z, t_span=torch.linspace(0, 1, 100, device=device))[-1]
        traj = traj.view(-1, 1, 128, 128).clamp(-1, 1) / 2 + 0.5  # Normalize to [0, 1]
    save_image(traj, f"{save_dir}{net}_generated_FM_images_step_{step}.png", nrow=4)
    model.train()

def ema(source_model, target_model, decay: float):
    for key, value in source_model.state_dict().items():
        target_model.state_dict()[key].data.mul_(decay).add_(value.data, alpha=1 - decay)

def infiniteloop(dataloader):
    while True:
        yield from dataloader