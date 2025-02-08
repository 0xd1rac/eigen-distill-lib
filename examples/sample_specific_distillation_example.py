#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from examples.utils import get_dataloaders, evaluate
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from distill_lib.items.student import Student
from distill_lib.items.teacher import Teacher
from distill_lib.strategies.weighting_strategies import SampleSpecificWeightingStrategy

def compute_teacher_performance(teacher_output: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the performance of a teacher on a batch.
    
    Args:
        teacher_output: Model predictions
        labels: Ground truth labels
        
    Returns:
        float: Accuracy score for the batch
    """
    _, predicted = torch.max(teacher_output.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get dataloaders for training and testing
    teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()
    
    # Set up multiple teacher models with different architectures
    teacher_models = []
    
    # ResNet50 teacher (good at texture-based features)
    resnet50_model = models.resnet50(pretrained=True)
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 10)
    teacher_models.append(Teacher(model=resnet50_model, data_loader=teacher_train_loader))
    
    # DenseNet121 teacher (good at fine-grained features)
    densenet_model = models.densenet121(pretrained=True)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 10)
    teacher_models.append(Teacher(model=densenet_model, data_loader=teacher_train_loader))
    
    # VGG16 teacher (good at shape-based features)
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, 10)
    teacher_models.append(Teacher(model=vgg_model, data_loader=teacher_train_loader))
    
    # Set up student model (smaller ResNet18)
    student_model = models.resnet18(pretrained=False)
    student_model.fc = nn.Linear(student_model.fc.in_features, 10)
    optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    student = Student(model=student_model, optimizer=optimizer, data_loader=student_train_loader)
    
    # Initialize the sample-specific weighting strategy with memory size
    weighting_strategy = SampleSpecificWeightingStrategy(memory_size=1000)
    
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
    
    print("Starting distillation with sample-specific teacher weighting...")
    print(f"Number of teachers: {len(teacher_models)}")
    print(f"Using device: {device}")
    print(f"Memory size for sample history: {weighting_strategy.memory_size}")
    
    # Pre-compute initial teacher performances on a few batches
    print("\nInitializing teacher performance history...")
    num_init_batches = 5
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(teacher_train_loader):
            if batch_idx >= num_init_batches:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            teacher_performances = []
            
            # Compute performance for each teacher
            for teacher in teacher_models:
                teacher.model.to(device)
                teacher.model.eval()
                outputs = teacher.model(inputs)
                performance = compute_teacher_performance(outputs, labels)
                teacher_performances.append(performance)
            
            # Update the weighting strategy's history
            weighting_strategy.update_history(inputs, teacher_performances)
            print(f"Batch {batch_idx + 1} teacher performances:", 
                  [f"{perf:.3f}" for perf in teacher_performances])
    
    # Training loop
    for epoch in range(num_epochs):
        running_losses = []
        
        # Use the distill method for training
        losses = distiller.distill(1, device, alpha=alpha, temperature=temperature)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Average losses: {losses}")
        
        # Update teacher performance history during training
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(teacher_train_loader):
                if batch_idx >= 10:  # Limit updates for example purposes
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                teacher_performances = []
                
                # Compute performance for each teacher
                for teacher in teacher_models:
                    teacher.model.eval()
                    outputs = teacher.model(inputs)
                    performance = compute_teacher_performance(outputs, labels)
                    teacher_performances.append(performance)
                
                # Update the weighting strategy's history
                weighting_strategy.update_history(inputs, teacher_performances)
    
    # Evaluate the student model
    student.model.eval()
    test_acc = evaluate(student.model, student_test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Print final teacher weights for a sample batch
    with torch.no_grad():
        sample_batch = next(iter(teacher_train_loader))[0].to(device)
        final_weights = weighting_strategy.compute_weights(teacher_models, sample_batch)
        print("\nFinal teacher weights for a sample batch:")
        for i, weight in enumerate(final_weights):
            print(f"Teacher {i + 1} weight: {weight:.3f}")
        
        # Print some statistics about the performance history
        if weighting_strategy.performance_history:
            avg_performances = [sum(p[i] for p in weighting_strategy.performance_history) / 
                              len(weighting_strategy.performance_history) 
                              for i in range(len(teacher_models))]
            print("\nAverage teacher performances in history:")
            for i, perf in enumerate(avg_performances):
                print(f"Teacher {i + 1} average performance: {perf:.3f}")

if __name__ == '__main__':
    main() 