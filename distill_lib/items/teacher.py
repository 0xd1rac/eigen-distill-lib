import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Teacher:
    def __init__(self, model: nn.Module, data_loader: DataLoader, device: str = 'cpu') -> None:
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)
