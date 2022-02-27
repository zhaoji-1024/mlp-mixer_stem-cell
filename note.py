import torch

losses = [0.3, 0.2, 0.6]
tensor = torch.tensor(losses).mean().detach().item()

print(type(tensor))