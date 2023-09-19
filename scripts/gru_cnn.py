"""
Deep GRU-CNN input 224 x 224
https://ieeexplore.ieee.org/abstract/document/9423965
https://discuss.pytorch.org/t/input-shape-to-gru-layer/171318
"""

import torch
import torch.nn as nn

class GRUCNN(nn.Module):
    def __init__(self, hidden_size=64, num_classes=3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 1,1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 1,1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1,1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 1,1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.gru_block = nn.Sequential(
          nn.GRUCell(512*7*7, hidden_size),
          nn.ReLU(),
          nn.BatchNorm1d(64),
          nn.Dropout(p=0.5)
        )

        self.fc= nn.Sequential(
            nn.Linear(64, num_classes), # 512*7*7*BATCH_SIZE
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.Dropout(p=0.5),
            nn.Softmax()
        )
        

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x) # torch.Size([32, 512, 7, 7])
        
        x = x.view(x.size(0), -1) # [32, 512 * 7 * 7]
        x = self.gru_block(x)
        x = self.fc(x)
        
        return x