import torch

model = torch.hub.load(
    "nicolvisser/WavTokenizer", "small_600_24k_4096", pretrained=True, progress=True
)

model.eval()

print(model)