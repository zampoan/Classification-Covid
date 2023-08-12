
""""
Original paper here: https://arxiv.org/pdf/1602.07360.pdf
"""

import torch.nn as nn

class FireModule(nn.Module):
    """
    s1x1: number of filters in squeeze layer (all 1x1)
    e1x1: number of 1x1 filters in expand layer
    e3x3: number of 3x3 filters in expand layer
    s1x1 > (e1x1 + e3x3)
    """
    def __init__(self, in_channels, s1x1,e1x1,e3x3,):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.expand_e1x1 = nn.Sequential(
            nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.expand_e3x3 = nn.Sequential(
            nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.squeeze(x)
        y = self.expand_e1x1(x)
        z = self.expand_e3x3(x)
        return torch.cat((y, z),dim=0)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.conv_10 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.maxpool_8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fireModule_2 = FireModule(96, 16, 64, 64)
        self.fireModule_3 = FireModule(128, 16, 64, 64)
        self.fireModule_4 = FireModule(128, 32, 128, 128)
        self.fireModule_5 = FireModule(256, 32, 128, 128)
        self.fireModule_6 = FireModule(256, 48, 192, 192)
        self.fireModule_7 = FireModule(384, 48, 192, 192)   
        self.fireModule_8 = FireModule(384, 64, 256, 256)
        self.fireModule_9 = FireModule(512, 64, 256, 256)
        self.avgpool_10 = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)



    def forward(self, x):
        x = self.conv_1(x) # [96, 111, 111]
        x = self.relu(x)
        x = self.maxpool_1(x) # [96, 55, 55]
        x = self.fireModule_2(x) # [128,55,55]
        x = self.fireModule_3(x) # [128,55,55]
        x = self.fireModule_4(x) # [256,55,55]
        x = self.maxpool_4(x) # [256,27,27]
        x = self.fireModule_5(x) # [256,27,27]
        x = self.fireModule_6(x) # [384, 27, 27]
        x = self.fireModule_7(x) # [384, 27, 27]
        x = self.fireModule_8(x) # [512, 27, 27]
        x = self.maxpool_8(x) # [512, 13, 13]
        x = self.fireModule_9(x) # [512, 13, 13]
        x = self.conv_10(x)
        x = self.relu(x)
        x = self.avgpool_10(x) # [3,1,1] 
        x = torch.squeeze(x, dim=1)
        x = x.reshape(1,3) # [1,3]

        return x
