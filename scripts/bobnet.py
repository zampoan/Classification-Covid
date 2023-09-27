"""
Takess input of 256 x 256
"""

import torch.nn
import torch

class BobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_3_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_3_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_6_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_6_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.block_7 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.block_1(x) 
        x = self.block_2(x) # [32, 16, 64, 64]
        x1 = self.block_3_1(x) 
        x2 = self.block_3_2(x) 
        x = torch.concat([x1,x2], dim=1) # [32, 64, 31, 31]
        x = self.block_4(x) # [32, 64, 16, 16]
        x = self.block_5(x) # [32, 64, 9, 9]
        x1 = self.block_6_1(x)
        x2 = self.block_6_2(x)
        x = torch.concat([x1, x2], dim=1) # [32, 256, 3, 3]
        x = self.block_7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
