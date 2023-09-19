"""
Contains code about CovidAid Model. Original paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9418407
"""
import torch
from torch import nn

class CovidAidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.covid_aid_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )
        self.covid_aid_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.covid_aid_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.covid_aid_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.covid_aid_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.covid_aid_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.covid_aid_7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU()
        )
        
        self.covid_aid_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.covid_aid_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.covid_aid_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.covid_aid_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(363, 3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.covid_aid_1(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_2(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_block_1(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_block_2(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_block_3(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_block_4(x)
        x = self.maxpool_1(x)
        x = self.covid_aid_3(x)
        x = self.covid_aid_4(x)
        x = self.covid_aid_5(x)
        x = self.covid_aid_6(x)
        x = self.covid_aid_7(x)
        x = self.flatten(x)
        x = self.linear(x) # [32,3]
        x = self.softmax(x)
        return x
