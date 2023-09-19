"""
Input 224 x 224
"""

class Block_B(nn.Module):
    def __init__(self, in_features, first_conv, second_conv):
        super().__init__()
        self.block_b = nn.Sequential(
            nn.Conv2d(in_features, first_conv,3),
            nn.BatchNorm2d(first_conv),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(first_conv, second_conv, 3),
            nn.BatchNorm2d(second_conv),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block_b(x)

class Block_C(nn.Module):
    def __init__(self, in_features, first_conv):
        super().__init__()
        self.block_c = nn.Sequential(
            nn.Conv2d(in_features, first_conv, 3),
            nn.BatchNorm2d(first_conv),
            nn.ReLU(),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block_c(x)
    
class Block_D(nn.Module):
    def __init__(self, in_features, first_conv):
        super().__init__()
        self.block_d = nn.Sequential(
            nn.Conv2d(in_features, first_conv, 3),
            nn.BatchNorm2d(first_conv),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block_d(x)


class STRMReNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_a = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2))
        
        self.block_b_1 = Block_B(64, 128, 128)
        self.block_c_1 = Block_C(64, 128)
        self.block_d_1 = Block_D(64, 128)

    def forward(self, x):
        x = self.block_a(x) # [32, 64, 110, 110]
        x0 = self.block_b_1(x) # [32, 128, 26, 26]
        x = self.block_c_1(x) # [32, 64, 110, 110]
        x2 = self.block_d_1(x) # [32, 128, 54, 54]

        return x