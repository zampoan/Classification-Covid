"""
Putting data into Imagefolder and Dataloader
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets

def create_dataloaders(train_dir, test_dir, gen_dir, transform, batch_size, num_workers):
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    gen_data = datasets.ImageFolder(root=gen_dir, transform=transform)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,batch_size=batch_size, num_workers=num_workers)
    gen_dataloader = DataLoader(dataset=gen_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    class_names = train_data.classes

    return train_dataloader, test_dataloader, gen_dataloader, class_names
