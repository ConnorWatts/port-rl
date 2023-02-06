import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def foward(self) -> None:
        ...