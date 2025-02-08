import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Teacher:
    def __init__(self, model: nn.Module, data_loader: DataLoader) -> None:
        self.model = model
        self.data_loader = data_loader
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
