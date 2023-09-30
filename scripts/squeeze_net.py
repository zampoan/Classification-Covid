""""
Squeeze Net takes input of 224 instead of 256, so have to scale it accordingly
"""
import torch
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
            nn.ReLU(inplace=True),
        )
        self.expand_e1x1 = nn.Sequential(
            nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.expand_e3x3 = nn.Sequential(
            nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        y = self.expand_e1x1(x) # 64
        z = self.expand_e3x3(x) # 64
        return torch.cat((y, z),dim=1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=0)
        self.conv_10 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.maxpool_8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.fireModule_2 = FireModule(96, 16, 64, 64)
        self.fireModule_3 = FireModule(128, 16, 64, 64)
        self.fireModule_4 = FireModule(128, 32, 128, 128)
        self.fireModule_5 = FireModule(256, 32, 128, 128)
        self.fireModule_6 = FireModule(256, 48, 192, 192)
        self.fireModule_7 = FireModule(384, 48, 192, 192)   
        self.fireModule_8 = FireModule(384, 64, 256, 256)
        self.fireModule_9 = FireModule(512, 64, 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool_10 = nn.AvgPool2d((1,1))
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(867, 3)



    def forward(self, x):
        x = self.conv_1(x) # [32, 96, 109, 109]
        x = self.relu(x)
        x = self.maxpool_1(x) # [32, 96, 54, 54]
        x = self.fireModule_2(x) # [32, 128,55,55]
        x = self.fireModule_3(x) # [128,55,55]
        x = self.fireModule_4(x) # [256,55,55]
        x = self.maxpool_4(x) # [256,27,27]
        x = self.fireModule_5(x) # [256,27,27]
        x = self.fireModule_6(x) # [384, 27, 27]
        x = self.fireModule_7(x) # [384, 27, 27]
        x = self.fireModule_8(x) # [512, 27, 27]
        x = self.maxpool_8(x) # [512, 13, 13]
        x = self.fireModule_9(x) # [512, 13, 13]
        x = self.dropout(x)
        x = self.conv_10(x)
        x = self.relu(x)
        x = self.avgpool_10(x) # [3,1,1] 
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x
