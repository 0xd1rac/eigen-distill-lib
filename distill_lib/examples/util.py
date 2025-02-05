import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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

def get_dataloader():
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
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    return train_loader, test_loader