import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
from model import CIFAR10Net
from dataset import CIFAR10Dataset
from transforms import get_transforms
from train import train, test
from torch.utils.data import DataLoader
from torchsummary import summary

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Calculate mean and std
    mean = trainset.data.mean(axis=(0,1,2))/255
    std = trainset.data.std(axis=(0,1,2))/255
    
    # Get transforms
    train_transform, test_transform = get_transforms(mean, std)
    
    # Create datasets
    train_dataset = CIFAR10Dataset(trainset.data, trainset.targets, train_transform)
    test_dataset = CIFAR10Dataset(testset.data, testset.targets, test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # Initialize model, criterion and optimizer
    model = CIFAR10Net().to(device)
    
    # Print model summary
    summary(model, input_size=(3, 32, 32))  # CIFAR10 images are 32x32 with 3 channels
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(1, 51):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main() 