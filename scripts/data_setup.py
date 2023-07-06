"""
Putting data into Imagefolder and Dataloader
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE=32
NUM_WORKERS=os.cpu_count()

def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_name = train_data.classes

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_names
