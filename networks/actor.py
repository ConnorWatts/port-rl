import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def foward(self) -> None:
        ...