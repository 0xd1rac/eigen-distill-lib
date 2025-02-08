#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Teacher datasets tend to have higher resolution than that of the student
    teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()
    
    # Set up multiple teacher models.
    teachers = [models.resnet50(pretrained=True) for _ in range(2)]
    for teacher in teachers:
        teacher.fc = nn.Linear(teacher.fc.in_features, 10)
        teacher = nn.DataParallel(teacher).to(device)  # Use DataParallel
    
    # Set up multiple student models.
    students = [models.resnet18(pretrained=False) for _ in range(3)]
    for student in students:
        student.fc = nn.Linear(student.fc.in_features, 10)
        student = nn.DataParallel(student).to(device)  # Use DataParallel
    
    # Set up optimizers for each student.
    optimizers = [optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for student in students]
    
    # Instantiate the SoftTargetDistiller with multiple students and teachers.
    distiller = SoftTargetDistiller(students, teachers, optimizers=optimizers)
    
    # Training hyperparameters.
    num_epochs = 1
    alpha = 0.5         # Weight for distillation loss.
    temperature = 2.0   # Temperature for softening outputs.

    # Use the distill method for training
    distiller.distill(teacher_train_loader, student_train_loader, num_epochs, device, alpha=alpha, temperature=temperature)
    
    # Evaluate each student model
    for i, student in enumerate(students):
        student.eval()  # Set the student model to evaluation mode
        test_acc = evaluate(student, student_test_loader, device)
        print(f"Student {i+1} Test Accuracy: {test_acc:.2f}%")
    
if __name__ == '__main__':
    main()
