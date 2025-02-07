#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.multilayer_activation_distiller import MultiLayerActivationDistiller

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders()

    # Set up the teacher model (larger model).
    teacher = models.resnet50(pretrained=True)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    teacher = teacher.to(device)

    # Set up the student model (smaller model).
    student = models.resnet18(pretrained=False)
    student.fc = nn.Linear(student.fc.in_features, 10)
    student = student.to(device)

    # Specify layers to hook
    student_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    teacher_layers = ['layer1', 'layer2', 'layer3', 'layer4']

    # Set up the optimizer and distiller.
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    distiller = MultiLayerActivationDistiller(student, teacher, optimizer=optimizer,
                                              student_layers=student_layers, teacher_layers=teacher_layers)

    # Training hyperparameters.
    num_epochs = 1
    alpha = 0.5  # Weight for activation loss.

    # Use the distill method for training
    distiller.distill(train_loader, num_epochs, device, alpha=alpha)

    # Evaluate the student model.
    test_acc = evaluate(student, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main() 