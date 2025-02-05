#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from examples.utils import *
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders()
    # --------------------------
    # Set up the teacher model.
    # --------------------------
    # Load a pretrained ResNet-50.
    teacher = models.resnet50(pretrained=True)
    # Replace the final fully connected layer to output 10 classes.
    num_ftrs_teacher = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs_teacher, 10)
    teacher = teacher.to(device)
    
    # ---------------------------
    # Set up the student model.
    # ---------------------------
    # Instantiate a ResNet-18 (from scratch or pretrained if desired).
    student = models.resnet18(pretrained=False)
    num_ftrs_student = student.fc.in_features
    student.fc = nn.Linear(num_ftrs_student, 10)
    student = student.to(device)
    
    # ---------------------------
    # Set up the optimizer and distiller.
    # ---------------------------
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Instantiate the SoftTargetDistiller.
    distiller = SoftTargetDistiller(student, teacher, optimizer=optimizer)
    
    # Training hyperparameters.
    num_epochs = 1
    alpha = 0.5         # Weight for distillation loss.
    temperature = 2.0   # Temperature for softening outputs.
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(distiller, train_loader, device, epoch, alpha, temperature)
        test_acc = evaluate(student, test_loader, device)
        print(f"Epoch {epoch}: Average Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    
if __name__ == '__main__':
    main()
