"""
Paper: https://www.sciencedirect.com/science/article/pii/S1568494622007050
"""
import torch
import torch.nn as nn

class Focus(nn.Module): 
    # Focus wh information into c-space 
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1):  # ch_in, ch_out, kernel, stride, padding, groups 
        super(Focus, self).__init__() 
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, p, groups=g) 
        # self.contract = Contract(gain=2) 

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2) 
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class CodnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 16)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.depthwise_conv_1 = nn.Sequential(
            nn.Conv2d(32, 28, 2, 2, groups=4), # kernel= 3x3?
            nn.Conv2d(28, 28, 1),
            nn.BatchNorm2d(28),
            nn.ReLU()
        )
        self.depthwise_conv_2 = nn.Sequential(
            nn.BatchNorm2d(44),
            nn.Hardswish(),
            nn.Conv2d(44, 128, 1),
            nn.Conv2d(128, 128, 5, 2, groups=4), # kernel=7x7?
            nn.Conv2d(128, 32, 1, padding=1)
        )
        self.depthwise_conv_3 = nn.Sequential(
            nn.BatchNorm2d(60),
            nn.Hardswish(),
            nn.Conv2d(60, 128, 1),
            nn.Conv2d(128, 128, 5, 2, groups=4), # kernel=7x7?
            nn.Conv2d(128, 32, 1, padding=1)
        )
        self.depthwise_conv_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, 128, 5, 2, groups=4), # kernel=7x7?
            nn.Conv2d(128, 32, 1, padding=1)
        )
        self.depthwise_conv_5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, 128, 5, 1, groups=4), # kernel=7x7?
            nn.Conv2d(128, 32, 1, padding=2)
        )
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.Hardswish(),
            nn.AvgPool2d(8)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(96, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(512, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
    
    def forward(self, x):
        # Initial Conv
        x_cv = self.conv_1(x) # [32, 16, 128, 128

        # Focus
        x_f= self.focus(x) # [32, 16, 128, 628]

        # Max Pool
        x_mp = self.maxpool(x_cv) # [32, 16, 64, 64]

        # Concat initial Conv with Focus
        x_cc1 = torch.concat((x_cv, x_f), dim=1) #[32, 32, 128, 128]

        # Depthwise conv 1
        x_dcv1= self.depthwise_conv_1(x_cc1) # [32, 28, 64, 64]

        # Avg Pool 1
        x_ap1 = self.avgpool(x_dcv1) # [32, 28, 32, 32]

        # Concat depwise conv with maxpool
        x_cc2 = torch.concat((x_dcv1, x_mp), dim=1) # [ 32, 44, 64, 64]
        
        # Depthwise conv 2
        x_dcv2 = self.depthwise_conv_2(x_cc2) #[32, 32, 32, 32]

        # Avg Pool 2
        x_ap2 = self.avgpool(x_dcv2) # [32, 32, 16, 16]

        # Concat depthwise conv 2 with avgpool 1
        x_cc3 = torch.concat((x_dcv2, x_ap1), dim=1) # [32, 60, 32, 32]

        # Deptwise conv 3
        x_dcv3 = self.depthwise_conv_3(x_cc3) # [32, 32, 16, 16]

        # Avg Pool 3
        x_ap3 = self.avgpool(x_dcv3) # [32,32,8,8]

        # Concat depthwise conv3 with avgpool 2
        x_cc4 = torch.concat((x_dcv3, x_ap2), dim=1) # [32, 64, 16, 16]

        # Depthwise conv 4
        x_dcv4 = self.depthwise_conv_4(x_cc4) # [32,32,8,8]

        # Concat depwise conv4 with avgpool 3
        x_cc5 = torch.concat((x_dcv4, x_ap3), dim=1) # [32, 64, 8, 8]

        # Depwise conv 5
        x_dcv5 = self.depthwise_conv_5(x_cc5) # [32, 32, 8, 8]

        # Concat Depthwise conv4, AvgPool 3 and x_cc5
        x = torch.cat((x_dcv4, x_ap3, x_dcv5), dim=1) # [32, 96, 8, 8]

        x = self.pool_block(x) # [32, 96, 1, 1]
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
