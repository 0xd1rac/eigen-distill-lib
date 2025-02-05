#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Import your SoftTargetDistiller class.
# Adjust the import below according to your project structure.
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller

def train_one_epoch(distiller, train_loader, device, epoch, alpha, temperature):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        loss = distiller.train_step(inputs, labels, alpha=alpha, temperature=temperature)
        running_loss += loss
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss:.4f}")
    return running_loss / len(train_loader)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data transforms for CIFAR-10. We resize images to 224x224 as ResNet models expect.
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),  # Standard normalization for ImageNet
                             (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    
    # CIFAR-10 datasets.
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    # Data loaders.
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
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
