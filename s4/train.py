import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Create directories for saving results
Path("static").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1: Input = 28x28x1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14x32
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 7x7x64
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7x128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 7x7x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 3x3x128
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 512),
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

def update_server(iteration, loss, accuracy, message):
    """Send updates to Flask server"""
    try:
        requests.post('http://localhost:5000/update_metrics', 
                     json={'iteration': iteration, 'loss': loss, 'accuracy': accuracy})
        requests.post('http://localhost:5000/update_log',
                     json={'message': message})
    except:
        print(f"Failed to update server: {message}")

def train():
    # Set device and print GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Data preprocessing
    print("Preparing datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)
    print(f"Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Initialize model, loss, and optimizer
    print("Initializing model...")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    update_server(0, 0, 0, "Starting training...")
    print("Beginning training...")
    
    # Training loop
    num_epochs = 1
    best_accuracy = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        # Use tqdm for the training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            # Update metrics every 50 batches
            if (i + 1) % 50 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / (i + 1)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'accuracy': f'{accuracy:.2f}%'
                })
                
                # Update server
                update_server(global_step, avg_loss, accuracy, 
                            f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            global_step += 1
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # Validation phase
        print("Running validation...")
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        message = f'Epoch [{epoch+1}/{num_epochs}] completed. Validation Accuracy: {val_accuracy:.2f}%'
        update_server(global_step, running_loss/len(train_loader), val_accuracy, message)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    print("Training completed. Generating final results...")
    
    # Load best model and generate final results
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Get 10 random test images
    random_indices = np.random.choice(len(test_dataset), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for idx, ax in enumerate(axes.flat):
        image, label = test_dataset[random_indices[idx]]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
        
        ax.imshow(image.cpu().squeeze(), cmap='gray')
        ax.axis('off')
        color = 'green' if pred == label else 'red'
        ax.set_title(f'True: {label}\nPred: {pred}', color=color)
    
    plt.tight_layout()
    plt.savefig('static/results.png')
    
    final_message = f"Training completed! Best accuracy: {best_accuracy:.2f}%"
    update_server(global_step, 0, best_accuracy, final_message)

if __name__ == '__main__':
    train()