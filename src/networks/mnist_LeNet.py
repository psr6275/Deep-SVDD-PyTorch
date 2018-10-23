import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4)

        self.code = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        self.conv3 = nn.Conv2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4)

        self.conv4 = nn.Conv2d(4, 8, 5, bias=False, padding=1)
        self.bn4 = nn.BatchNorm2d(8)

        self.conv5 = nn.Conv2d(8, 1, 5, bias=False, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.code(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.conv3(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.conv4(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x