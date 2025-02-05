#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

# Import your MultiLayerActivationDistiller class.
# Adjust the import below according to your project structure.
from distill_lib.offline.multilayer_activation_distiller import MultiLayerActivationDistiller

# Define a dummy model that returns (logits, activations_dict).
class DummyMultiActivationModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(DummyMultiActivationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Assume input images are 32x32.
        self.fc = nn.Linear(hidden_channels * 2 * 32 * 32, num_classes)
        
    def forward(self, x):
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        logits = self.fc(self.flatten(act2))
        # Return a dictionary of activations.
        activations = {'layer1': act1, 'layer2': act2}
        return logits, activations

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Instantiate teacher and student models.
    teacher = DummyMultiActivationModel(in_channels=3, hidden_channels=16, num_classes=10).to(device)
    student = DummyMultiActivationModel(in_channels=3, hidden_channels=8, num_classes=10).to(device)
    
    # Create an optimizer for the student.
    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    
    # Instantiate the MultiLayerActivationDistiller.
    # Here, we do not provide an explicit alignment map, so the distiller will automatically align
    # the activation layers by sorted key order. (e.g., 'layer1' with 'layer1' and 'layer2' with 'layer2')
    distiller = MultiLayerActivationDistiller(student, teacher, optimizer=optimizer)
    
    # Create dummy data.
    inputs = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    # Perform one training step.
    loss = distiller.train_step(inputs, labels, alpha=0.5)
    print("Multi-Layer Activation Distillation Training Step Loss:", loss)

if __name__ == '__main__':
    main()
