from torch.utils.data import Dataset
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, target 