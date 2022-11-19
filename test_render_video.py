import torch

from render import RenderBot

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

renderbot = RenderBot()
renderbot.render_from_file(input_fn = "Robot4_dt0.0001.pt",
                           result_fn = "Robot4_dt0.0001_500fps.mp4",
                           fps = 500, every = 100)
