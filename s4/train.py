import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
import requests
import argparse
import json
from pathlib import Path

# Create directories
Path("static").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channels['layer1'], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels['layer1']),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels['layer1'], channels['layer2'], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels['layer2']),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels['layer2'], channels['layer3'], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels['layer3']),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels['layer3'], channels['layer4'], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels['layer4']),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(channels['layer4'] * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def update_metrics(iteration, loss, accuracy, model_id, message, is_validation=False, current_epoch=0):
    """Update metrics - will only send to server if server mode is enabled"""
    if not hasattr(update_metrics, 'server_mode') or not update_metrics.server_mode:
        return
        
    try:
        # Print to console for debugging
        print(f"Sending metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Epoch: {current_epoch}")
        
        data = {
            'iteration': int(iteration),
            'loss': float(loss),
            'accuracy': float(accuracy),
            'model_id': int(model_id),
            'is_validation': bool(is_validation),
            'current_epoch': int(current_epoch),
            'message': str(message)
        }
        response = requests.post('http://localhost:5000/update_metrics', json=data)
        if response.status_code != 200:
            print(f"Server returned error: {response.text}")
    except Exception as e:
        print(f"Failed to update server: {str(e)}")

def train_model(model_id, channels, batch_size, epochs, optimizer_config=None, results=None, training_log=None):
    """Train a single model with the specified configuration"""
    try:
        # Set device and log CUDA status
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cuda_status = f"CUDA {'is' if torch.cuda.is_available() else 'is not'} available"
        if torch.cuda.is_available():
            cuda_status += f" (Using {torch.cuda.get_device_name(0)})"
        print(f"[Model {model_id}] {cuda_status}")
        
        # Data loading and preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print(f"[Model {model_id}] Loading MNIST dataset...")
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        print(f"[Model {model_id}] Setting up data loaders with batch size: {batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"[Model {model_id}] Initializing model...")
        model = CNN(channels).to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer based on configuration
        if optimizer_config is None:
            optimizer_config = {'name': 'adam', 'learning_rate': 0.001}
        
        if optimizer_config['name'].lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), 
                                lr=optimizer_config['learning_rate'],
                                momentum=0.9)
        elif optimizer_config['name'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), 
                                    lr=optimizer_config['learning_rate'])
        elif optimizer_config['name'].lower() == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), 
                                    lr=optimizer_config['learning_rate'])
        else:  # default to Adam
            optimizer = optim.Adam(model.parameters(), 
                                 lr=optimizer_config['learning_rate'])
        
        best_accuracy = 0
        global_step = 0
        total_batches = len(train_loader)
        
        print(f"[Model {model_id}] Starting training for {epochs} epochs...")
        # Main training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Create progress bar for batches
            batch_pbar = tqdm(enumerate(train_loader), 
                            total=len(train_loader),
                            desc=f'Model {model_id} Epoch {epoch+1}/{epochs}',
                            leave=True)
            
            # Batch training loop
            for i, (images, labels) in batch_pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch metrics
                batch_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                
                # Update running metrics
                running_loss = (running_loss * i + batch_loss) / (i + 1)  # Running average
                total += batch_total
                correct += batch_correct
                accuracy = 100 * correct / total
                
                # Calculate progress percentage
                progress = (i + 1) / total_batches * 100
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'acc': f'{accuracy:.2f}%',
                    'progress': f'{progress:.1f}%'
                })
                
                # Update server with immediate batch results
                update_metrics(
                    global_step,
                    running_loss,
                    accuracy,
                    model_id,
                    f"Model {model_id} - Epoch {epoch+1}/{epochs}, Progress: {progress:.1f}%",
                    current_epoch=epoch+1
                )
                
                global_step += 1
            
            # Validation phase after each epoch
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Create progress bar for validation
            val_pbar = tqdm(test_loader, 
                          desc=f'Model {model_id} Validation',
                          leave=True)
            
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Calculate running validation metrics
                    current_val_loss = val_loss / (val_total / labels.size(0))
                    current_val_acc = 100 * val_correct / val_total
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({
                        'loss': f'{current_val_loss:.4f}',
                        'acc': f'{current_val_acc:.2f}%'
                    })
                    
                    # Update server with validation progress
                    update_metrics(
                        global_step,
                        current_val_loss,
                        current_val_acc,
                        model_id,
                        f"Model {model_id} - Validation Epoch {epoch+1}/{epochs}",
                        is_validation=True,
                        current_epoch=epoch+1
                    )
            
            # Calculate final validation metrics
            val_loss = val_loss / len(test_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Save best model if needed
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'config': {
                        'channels': channels,
                        'batch_size': batch_size,
                        'epochs': epochs
                    }
                }, f'model_{model_id}_best.pth')
                print(f"[Model {model_id}] New best model saved with accuracy: {best_accuracy:.2f}%")
        
        return best_accuracy

    except Exception as e:
        error_message = f"Error training Model {model_id}: {str(e)}"
        print(error_message)
        update_metrics(0, 0, 0, model_id, error_message, current_epoch=0)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST CNN Model')
    parser.add_argument('--model-id', type=int, default=1, help='Model ID (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--server-mode', action='store_true', help='Enable server updates')
    
    # Add channel arguments
    parser.add_argument('--layer1', type=int, default=32, help='Channels in layer 1')
    parser.add_argument('--layer2', type=int, default=64, help='Channels in layer 2')
    parser.add_argument('--layer3', type=int, default=128, help='Channels in layer 3')
    parser.add_argument('--layer4', type=int, default=256, help='Channels in layer 4')
    
    args = parser.parse_args()
    
    # Configure channels
    channels = {
        'layer1': args.layer1,
        'layer2': args.layer2,
        'layer3': args.layer3,
        'layer4': args.layer4
    }
    
    # Set server mode
    update_metrics.server_mode = args.server_mode
    
    print("\nTraining Configuration:")
    print(f"Model ID: {args.model_id}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Channels: {channels}")
    print(f"Server Mode: {'Enabled' if args.server_mode else 'Disabled'}\n")
    
    try:
        accuracy = train_model(
            model_id=args.model_id,
            channels=channels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            results={},
            training_log=[]
        )
        print(f"\nTraining completed successfully! Final accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)