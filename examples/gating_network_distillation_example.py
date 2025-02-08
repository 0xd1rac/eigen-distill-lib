#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from distill_lib.items.student import Student
from distill_lib.items.teacher import Teacher
from distill_lib.strategies.weighting_strategies import GatingNetworkWeightingStrategy

def get_feature_size(model):
    """Helper function to get the feature size of the model's penultimate layer."""
    if isinstance(model, models.ResNet):
        return model.fc.in_features
    elif isinstance(model, models.DenseNet):
        return model.classifier.in_features
    elif isinstance(model, models.VGG):
        return model.classifier[-1].in_features
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get dataloaders for training and testing
    teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()
    
    # Set up multiple teacher models with different architectures
    teacher_models = []
    
    # ResNet50 teacher
    resnet50_model = models.resnet50(pretrained=True)
    feature_size = get_feature_size(resnet50_model)  # Get feature size from first teacher
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 10)
    teacher_models.append(Teacher(model=resnet50_model, data_loader=teacher_train_loader))
    
    # DenseNet121 teacher
    densenet_model = models.densenet121(pretrained=True)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 10)
    teacher_models.append(Teacher(model=densenet_model, data_loader=teacher_train_loader))
    
    # VGG16 teacher (adding a third teacher for more diversity)
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, 10)
    teacher_models.append(Teacher(model=vgg_model, data_loader=teacher_train_loader))
    
    # Set up student model (smaller ResNet18)
    student_model = models.resnet18(pretrained=False)
    student_model.fc = nn.Linear(student_model.fc.in_features, 10)
    optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    student = Student(model=student_model, optimizer=optimizer, data_loader=student_train_loader)
    
    # Initialize the gating network weighting strategy
    weighting_strategy = GatingNetworkWeightingStrategy(
        feature_size=feature_size,  # Use the feature size from the penultimate layer
        num_teachers=len(teacher_models)
    )
    
    # Move gating network to the same device
    weighting_strategy.gating_network.to(device)
    
    # Create optimizer for the gating network
    gating_optimizer = optim.Adam(weighting_strategy.gating_network.parameters(), lr=0.001)
    
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
    
    print("Starting distillation with gating network teacher weighting...")
    print(f"Number of teachers: {len(teacher_models)}")
    print(f"Using device: {device}")
    print(f"Feature size for gating network: {feature_size}")
    
    # Training loop with gating network updates
    for epoch in range(num_epochs):
        # Use the distill method for training
        losses = distiller.distill(1, device, alpha=alpha, temperature=temperature)
        
        # Update gating network
        gating_optimizer.step()
        gating_optimizer.zero_grad()
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Average losses: {losses}")
    
    # Evaluate the student model
    student.model.eval()  # Set the student model to evaluation mode
    test_acc = evaluate(student.model, student_test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Print final teacher weights for a sample batch
    with torch.no_grad():
        sample_batch = next(iter(teacher_train_loader))[0].to(device)
        final_weights = weighting_strategy.compute_weights(teacher_models, sample_batch)
        print("\nFinal teacher weights for a sample batch:")
        for i, weight in enumerate(final_weights):
            print(f"Teacher {i + 1} weight: {weight:.3f}")

if __name__ == '__main__':
    main() 