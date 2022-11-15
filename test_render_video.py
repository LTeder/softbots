import torch

from render import RenderBot

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

renderbot = RenderBot()
renderbot.render_from_file(input_fn = "RandomSearchRobot.pt",
                           result_fn = "RandomSearchRobot_1000fps.mp4",
                           fps = 1000, every = 100)
