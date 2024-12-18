import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(mean, std):
    # Convert numpy arrays to lists if needed
    mean = mean.tolist() if isinstance(mean, np.ndarray) else mean
    std = std.tolist() if isinstance(std, np.ndarray) else std
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=16,
            min_holes=1, min_height=16, min_width=16,
            fill_value=0, p=0.2
        ),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform 