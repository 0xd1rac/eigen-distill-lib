#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from distill_lib.items.student import Student
from distill_lib.items.teacher import Teacher

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Teacher datasets tend to have higher resolution than that of the student
    teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()
    
    # Set up multiple teacher models
    teacher_models = []
    for _ in range(2):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
        teacher_models.append(Teacher(model=model, device=device, data_loader=teacher_train_loader))
    
    # Set up multiple student models
    student_models = []
    for _ in range(3):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        student_models.append(Student(model=model, optimizer=optimizer, device=device, data_loader=student_train_loader))
    
    # Instantiate the SoftTargetDistiller with multiple students and teachers
    distiller = SoftTargetDistiller(students=student_models, teachers=teacher_models)
    
    # Training hyperparameters
    num_epochs = 1
    alpha = 0.5         # Weight for distillation loss
    temperature = 2.0   # Temperature for softening outputs

    # Use the distill method for training (no need to pass data loaders as they're in the Student/Teacher instances)
    distiller.distill(num_epochs=num_epochs, alpha=alpha, temperature=temperature)
    
    # Evaluate each student model
    for i, student in enumerate(student_models):
        student.model.eval()  # Set the student model to evaluation mode
        test_acc = evaluate(student.model, student_test_loader, device)
        print(f"Student {i+1} Test Accuracy: {test_acc:.2f}%")
    
if __name__ == '__main__':
    main()
