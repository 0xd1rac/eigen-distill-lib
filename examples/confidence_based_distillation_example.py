#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from distill_lib.items.student import Student
from distill_lib.items.teacher import Teacher
from distill_lib.strategies.weighting_strategies import ConfidenceBasedWeightingStrategy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get dataloaders for training and testing
    teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()
    
    # Set up multiple teacher models with different architectures
    teacher_models = []
    
    # ResNet50 teacher
    resnet50_model = models.resnet50(pretrained=True)
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 10)
    teacher_models.append(Teacher(model=resnet50_model, data_loader=teacher_train_loader))
    
    # DenseNet121 teacher (different architecture for diversity)
    densenet_model = models.densenet121(pretrained=True)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 10)
    teacher_models.append(Teacher(model=densenet_model, data_loader=teacher_train_loader))
    
    # Set up student model (smaller ResNet18)
    student_model = models.resnet18(pretrained=False)
    student_model.fc = nn.Linear(student_model.fc.in_features, 10)
    optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    student = Student(model=student_model, optimizer=optimizer, data_loader=student_train_loader)
    
    # Initialize the confidence-based weighting strategy
    weighting_strategy = ConfidenceBasedWeightingStrategy()
    
    # Instantiate the SoftTargetDistiller with the weighting strategy
    distiller = SoftTargetDistiller(
        students=[student],
        teachers=teacher_models,
        weighting_strategy=weighting_strategy
    )
    
    # Training hyperparameters
    num_epochs = 1
    alpha = 0.5         # Weight for distillation loss
    temperature = 2.0   # Temperature for softening outputs
    
    print("Starting distillation with confidence-based teacher weighting...")
    print(f"Number of teachers: {len(teacher_models)}")
    print(f"Using device: {device}")
    
    # Use the distill method for training
    losses = distiller.distill(num_epochs, device, alpha=alpha, temperature=temperature)
    
    # Evaluate the student model
    student.model.eval()  # Set the student model to evaluation mode
    test_acc = evaluate(student.model, student_test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("\nAverage losses during training:", losses)

if __name__ == '__main__':
    main() 