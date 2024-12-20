import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.model1 import Net as Model1
from utils.train import train
from utils.test import test
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
import os

def get_model(model_name):
    models = {
        'model1': Model1
    }
    return models.get(model_name, Model1)

def plot_training_curves(train_metrics, test_metrics, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(test_metrics, label=f'Test {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs. Epochs')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main(model_name='model1'):
    # Training settings
    EPOCHS = 20
    SEED = 1
    
    # CUDA settings
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    # Data transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dataset and loaders
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
    
    # Model
    ModelClass = get_model(model_name)
    model = ModelClass().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    
    # Add model summary
    print(f"\nModel Summary for {model_name}:")
    summary(model, input_size=(1, 1, 28, 28))
    
    # Lists to store metrics
    train_losses_history = []
    train_acc_history = []
    test_losses_history = []
    test_acc_history = []
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        train_losses, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_losses, test_acc = test(model, device, test_loader)
        scheduler.step()
        
        # Store metrics
        train_losses_history.append(np.mean([loss.detach().cpu().numpy() if torch.is_tensor(loss) else loss for loss in train_losses]))
        train_acc_history.append(train_acc)
        test_losses_history.append(np.mean([loss.detach().cpu().numpy() if torch.is_tensor(loss) else loss for loss in test_losses]))
        test_acc_history.append(test_acc)
    
    # Generate and save plots
    plots_dir = f"plots/{model_name}"
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_training_curves(
        train_losses_history, 
        test_losses_history, 
        'Loss', 
        f'{plots_dir}/loss_curve.png'
    )
    
    plot_training_curves(
        train_acc_history, 
        test_acc_history, 
        'Accuracy', 
        f'{plots_dir}/accuracy_curve.png'
    )
    
    # Save final metrics
    final_metrics = {
        'final_train_loss': train_losses_history[-1],
        'final_train_acc': train_acc_history[-1],
        'final_test_loss': test_losses_history[-1],
        'final_test_acc': test_acc_history[-1],
    }
    
    with open(f'{plots_dir}/metrics.txt', 'w') as f:
        for metric, value in final_metrics.items():
            f.write(f'{metric}: {value}\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model1', choices=['model1', 'model2', 'model3'])
    args = parser.parse_args()
    main(args.model)