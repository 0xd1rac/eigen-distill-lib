import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Student:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, device: str = 'cpu') -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device

        self.model = self.model.to(self.device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)
