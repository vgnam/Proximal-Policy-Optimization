import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class Actor(nn.Module):
    def __init__(self, input_channels=3, action_size=4):
        super(Actor, self).__init__()
        self.conv1 = self.block(input_channels, 32)
        self.conv2 = self.block(32, 64)
        self.conv3 = self.block(64, 128)
        self.conv4 = self.block(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Normalize activations in the final layer (output logits)
        x = F.relu(x)  # Applying ReLU after final layer
        distribution = Categorical(logits=x)
        return distribution


# Critic Network
class Critic(nn.Module):
    def __init__(self, input_channels=3):
        super(Critic, self).__init__()
        self.conv1 = self.block(input_channels, 32)
        self.conv2 = self.block(32, 64)
        self.conv3 = self.block(64, 64)
        self.conv4 = self.block(64, 128)
        self.conv5 = self.block(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


