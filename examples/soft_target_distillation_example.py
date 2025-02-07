#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders()
    
    # --------------------------
    # Set up the teacher model.
    # --------------------------
    teacher = models.resnet50(pretrained=True)
    num_ftrs_teacher = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs_teacher, 10)
    teacher = teacher.to(device)
    
    # ---------------------------
    # Set up the student model.
    # ---------------------------
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

    # Use the distill method for training
    distiller.distill(train_loader, num_epochs, device, alpha=alpha, temperature=temperature)
    
    # Evaluate the student model
    test_acc = evaluate(student, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
if __name__ == '__main__':
    main()
