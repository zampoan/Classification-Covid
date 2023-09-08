"""
Deep GRU-CNN input 224 x 224
https://ieeexplore.ieee.org/abstract/document/9423965
https://discuss.pytorch.org/t/input-shape-to-gru-layer/171318
"""

import torch
import torch.nn as nn

class GRUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 64, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 1,1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 1,1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 1,1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(512*7*7*BATCH_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax()
        )
        # self.gru = nn.GRU(input_size=3584, hidden_size=512, num_layers=1, batch_first=True)
        self.gru = nn.GRU(input_size=1, hidden_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x) # torch.Size([32, 512, 7, 7])
        
        # reshape the tensor 
        x = x.view(1, -1, 1) # [1, 512*7*7*BATCH_SIZE, 1]
        x, _ = self.gru(x) # _ represents hidden state
        x = self.relu(x)

        x = torch.flatten(x)
        x = self.dense(x)
        x = torch.unsqueeze(x,dim=0)
        
        return x
