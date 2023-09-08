"""
Paper: https://www.sciencedirect.com/science/article/pii/S1568494622007050#fig3
Efficient_CNN: https://www.hindawi.com/journals/complexity/2021/6621607/fig4/

"""
import torch.nn as nn

class EFFICIENT_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.block_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.block_5 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.block_6 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.MaxPool2d(1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.dense_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.dense_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.dense_3 = nn.Sequential(
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = torch.flatten(x)

        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = torch.unsqueeze(x, dim=0)
        return x
