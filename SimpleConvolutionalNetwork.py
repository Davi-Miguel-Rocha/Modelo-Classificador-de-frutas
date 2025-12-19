

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import transforms

from torch.utils.data import DataLoader


transform = transforms.Compose([

    transforms.ToTensor()
])









class SimpleConvolutionalNetwork(nn.Module):

    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(

            in_channels = 3,

            out_channels= 8,

            kernel_size= 3
        )

        self.conv2 = nn.Conv2d(

            in_channels = 3,

            out_channels= 8,

            kernel_size= 3
        )

        self.fc1 = nn.Linear(16 * 5 * 5, 64)

        self.fc2 = nn.Linear(64, 10)

        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):

        x = torch.relu(self.conv(x))

        x = self.pool(x)


        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x,1)

        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x