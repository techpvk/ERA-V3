# CIFAR10 Image Classification with Custom CNN

This project implements a custom CNN architecture for CIFAR10 image classification using PyTorch. The model features depthwise separable convolutions, dilated convolutions, and follows a C1C2C3C40 architecture pattern.

## Project Structure
project/
│
├── model.py # Model architecture definition
├── dataset.py # Custom dataset class
├── transforms.py # Albumentations transformations
├── train.py # Training and testing functions
├── main.py # Main execution file
└── README.md

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- albumentations
- tqdm
- torchsummary

Install the required packages using:

```bash
pip install torch torchvision albumentations numpy tqdm
```

## Model Architecture

The model follows a C1C2C3C40 architecture with:
- No MaxPooling layers
- Depthwise Separable Convolution in C2
- Dilated Convolution in C3
- Strided Convolution in C4
- Global Average Pooling
- Final FC layer

Key features:
- Total parameters: < 200k
- Receptive Field: > 44
- Target accuracy: 85%

## Data Augmentation

Using Albumentations library with:
- Horizontal Flip
- ShiftScaleRotate
- CoarseDropout with:
  - max_holes = 1
  - max_height = 16px
  - max_width = 16px
  - min_holes = 1
  - min_height = 16px
  - min_width = 16px

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training:

```bash
python main.py
```

## Code Structure

### model.py
Contains the model architecture with custom layers including:
- DepthwiseSeparableConv class
- CIFAR10Net class

### dataset.py
Custom dataset class for CIFAR10 with Albumentations support

### transforms.py
Transformation pipeline using Albumentations library

### train.py
Training and testing loops with progress tracking

### main.py
Main execution file that brings everything together

## Training Logs

The training progress is displayed with:
- Per epoch training loss and accuracy
- Validation loss and accuracy after each epoch
- Progress bar with real-time metrics

## Model Performance

The model achieves:
- Training accuracy: ~XX%
- Validation accuracy: ~XX%
- Total parameters: XXX,XXX

## Results Visualization

Training metrics are displayed during training:
```
Epoch: 1 Loss: X.XXXX Accuracy: XX.XX%
Test set: Average loss: X.XXXX, Accuracy: XXXX/10000 (XX.XX%)
...
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
