import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset

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

def get_dataloaders():
    # Define transformations for the teacher and student datasets
    teacher_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    student_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load datasets
    teacher_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=teacher_transform)
    student_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=student_transform)
    teacher_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=teacher_transform)
    student_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=student_transform)

    num_samples = 1
    indices = list(range(num_samples))
    teacher_train_dataset = Subset(teacher_train_dataset, indices)
    student_train_dataset = Subset(student_train_dataset, indices)
    teacher_test_dataset = Subset(teacher_test_dataset, indices)
    student_test_dataset = Subset(student_test_dataset, indices)

    # Create data loaders
    teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=1, shuffle=True, num_workers=2)
    student_train_loader = DataLoader(student_train_dataset, batch_size=1, shuffle=True, num_workers=2)
    teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=1, shuffle=False, num_workers=2)
    student_test_loader = DataLoader(student_test_dataset, batch_size=1, shuffle=False, num_workers=2)

    return teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader