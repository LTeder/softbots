import torch
from pathlib import Path

from renderbot import RenderBot

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

fn = Path("Crossed3DCube_Points_time.pt")
assert fn.exists()

#renderbot = RenderBot()
RenderBot().render_from_file(input_fn = fn, result_fn = "Crossed3DCube.mp4", fps = 500)
